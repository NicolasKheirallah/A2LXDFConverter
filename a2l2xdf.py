import argparse
import io
import logging
import re
import uuid
from collections import Counter
from typing import Any, Dict, List, Set, Optional, Tuple, Callable

# Prefer the modern package name (pya2ldb). Older 'pya2l' 0.1.x lacks DB.
try:  # pragma: no cover - import guard
    from pya2l import DB
    import pya2l.model as model
    from pya2l.api import inspect
except ImportError as exc:  # pragma: no cover - user environment guard
    raise ImportError(
        "pya2l.DB not found. Install the maintained package with:"
        " pip install pya2ldb"
    ) from exc

from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, ElementTree

import importlib.metadata
import pathlib

BASE_OFFSET = 0
RAM_BASE_OFFSET = 0  # For MEASUREMENT addresses (RAM)
SEGMENTS: List[Any] = []
categories: List[str] = []
axis_addresses_in_xdf: Set[str] = set()
data_addresses_in_xdf: Set[str] = set()
address_mappings: List[Dict[str, int]] = []

# XDF EMBEDDEDDATA type flags (TunerPro specification)
XDF_FLAG_SIGNED = 0x01      # Signed integer or float
XDF_FLAG_AXIS = 0x02        # Axis data (vs table data)
XDF_FLAG_TABLE = 0x04       # Table data (vs axis data)
XDF_FLAG_FLOAT = 0x10000    # IEEE floating point

# Default memory region size
DEFAULT_REGION_SIZE = 0x400000  # 4MB

# Default ADX poll rate
DEFAULT_ADX_POLL_RATE_MS = 100  # milliseconds

# Axis resolution recursion depth limit
MAX_AXIS_RESOLUTION_DEPTH = 2

# --------------------------------------------------------------------------- #
# Validation and Diagnostics                                                  #
# --------------------------------------------------------------------------- #

class ValidationReport:
    """Collects validation warnings, errors, and info during conversion."""
    
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.info = []
        self.address_map = {}  # address -> list of characteristic names
        
    def add_warning(self, msg: str):
        """Add a warning message."""
        self.warnings.append(msg)
        logging.warning(msg)
    
    def add_error(self, msg: str):
        """Add an error message."""
        self.errors.append(msg)
        logging.error(msg)
    
    def add_info(self, msg: str):
        """Add an info message."""
        self.info.append(msg)
    
    def check_address_overlap(self, name: str, address: str, size: int):
        """Check if address overlaps with existing characteristics."""
        if not address:
            return
            
        try:
            addr_int = int(address, 16) if isinstance(address, str) else address
        except (ValueError, TypeError):
            return
        
        # Check for exact address match
        if addr_int in self.address_map:
            existing = self.address_map[addr_int]
            self.add_warning(
                f"Address overlap: {name} shares address {hex(addr_int)} "
                f"with {', '.join(existing)}"
            )
            existing.append(name)
        else:
            self.address_map[addr_int] = [name]
    
    def print_summary(self):
        """Print validation summary to console."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for err in self.errors[:10]:  # Show first 10
                print(f"  - {err}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warn in self.warnings[:10]:
                print(f"  - {warn}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")
        
        if self.info:
            print(f"\n✓ INFO:")
            for inf in self.info:
                print(f"  - {inf}")
        
        print("="*60 + "\n")

# Global validation report instance
validation_report = ValidationReport()

# --------------------------------------------------------------------------- #
# Helpers for record layouts and endian/padding                               #
# --------------------------------------------------------------------------- #

def get_byte_order(obj) -> str:
    """Return declared byte order if present; default little-endian."""
    for attr in ("byteOrder", "byteorder", "BYTE_ORDER", "byte_order"):
        v = getattr(obj, attr, None)
        if v:
            return str(v)
    return "LSB_FIRST"


def is_little_endian(obj) -> bool:
    bo = get_byte_order(obj).upper()
    return "MSB" not in bo


def get_address_offset(obj) -> int:
    for attr in ("addressOffset", "address_offset", "ADDRESS_OFFSET"):
        v = getattr(obj, attr, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return 0


def get_dist_op_stride(axis_layout) -> Optional[int]:
    """
    If the axis layout has DIST_OP / data_size fields that imply a major stride,
    return stride in bytes.
    """
    if axis_layout is None:
        return None
    dist = getattr(axis_layout, "dist_op", None) or getattr(axis_layout, "DIST_OP", None)
    dsize = getattr(axis_layout, "data_size", None)
    if dist is None or dsize is None:
        return None
    try:
        dist = int(dist)
        dsize = int(dsize)
        if dist > 0 and dsize > 0:
            return dist * dsize
    except Exception:
        return None
    return None


def get_alignment_bits(record_layout) -> int:
    """
    Return alignment in bits if present; otherwise 0 (no extra padding).
    Considers ALIGNMENT, ALIGNMENT_BYTE / WORD / LONG if present.
    """
    if record_layout is None:
        return 0
    for attr in ("alignment", "ALIGNMENT"):
        v = getattr(record_layout, attr, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    # compatibility shorthands
    if getattr(record_layout, "ALIGNMENT_BYTE", None):
        return 8
    if getattr(record_layout, "ALIGNMENT_WORD", None):
        return 16
    if getattr(record_layout, "ALIGNMENT_LONG", None):
        return 32
    return 0


def get_fnc_values_index_mode(record_layout) -> str:
    """
    Extract the FNC_VALUES indexMode from record layout.

    Returns:
        "COLUMN_DIR": Column-major order (X varies fastest)
        "ROW_DIR": Row-major order (Y varies fastest)
        "ALTERNATE_CURVES": Alternating curve storage
        "ALTERNATE_WITH_X": X-axis stored with each row
        "ALTERNATE_WITH_Y": Y-axis stored with each column
        None: Unknown/not specified (default to ROW_DIR)

    According to ASAM MCD-2 MC:
    - COLUMN_DIR: Z[y][x] iterated as Z[0,0], Z[1,0], ..., Z[rows-1,0], Z[0,1], Z[1,1], ...
    - ROW_DIR: Z[y][x] iterated as Z[0,0], Z[0,1], ..., Z[0,cols-1], Z[1,0], Z[1,1], ...
    """
    if record_layout is None:
        return None

    # Try to get FNC_VALUES attribute
    fnc_values = getattr(record_layout, "fncValues", None) or getattr(record_layout, "fnc_values", None)
    if fnc_values is None:
        return None

    # Extract indexMode - can be attribute or dict-like
    index_mode = None
    for attr in ("indexMode", "index_mode", "IndexMode"):
        v = getattr(fnc_values, attr, None)
        if v is not None:
            index_mode = str(v).upper()
            break

    # Return normalized value
    if index_mode and "COLUMN" in index_mode:
        return "COLUMN_DIR"
    elif index_mode and "ROW" in index_mode:
        return "ROW_DIR"
    elif index_mode and "ALTERNATE" in index_mode:
        if "CURVES" in index_mode:
            return "ALTERNATE_CURVES"
        elif "X" in index_mode:
            return "ALTERNATE_WITH_X"
        elif "Y" in index_mode:
            return "ALTERNATE_WITH_Y"

    return None


def align_address(addr: int, alignment_bits: int) -> int:
    """Align address upward to alignment_bits boundary (bits)."""
    if alignment_bits <= 0:
        return addr
    mask = (alignment_bits // 8) - 1
    return (addr + mask) & ~mask


class LayoutComponent:
    def __init__(self, name, position, datatype, relation):
        self.name = name
        self.position = int(position)
        self.datatype = str(datatype)
        self.relation = relation  # 'NO_AXIS', 'AXIS_PTS', 'FNC_VALUES'

    def __repr__(self):
        return f"<LayoutComponent {self.name} pos={self.position} type={self.datatype}>"


def parse_record_layout(rl) -> List[LayoutComponent]:
    """
    Parse a pya2l RecordLayout object into a sorted list of components.
    """
    components = []
    if rl is None:
        return components

    # Map of attribute name -> relation type
    # Note: pya2l uses snake_case for attributes
    attrs = {
        "no_axis_pts_x": "NO_AXIS",
        "no_axis_pts_y": "NO_AXIS",
        "no_axis_pts_z": "NO_AXIS",
        "no_axis_pts_4": "NO_AXIS",
        "no_axis_pts_5": "NO_AXIS",
        "axis_pts_x": "AXIS_PTS",
        "axis_pts_y": "AXIS_PTS",
        "axis_pts_z": "AXIS_PTS",
        "axis_pts_4": "AXIS_PTS",
        "axis_pts_5": "AXIS_PTS",
        "fnc_values": "FNC_VALUES",
    }

    # 1. Check direct attributes (covers fncValues and flat layouts)
    for attr, relation in attrs.items():
        val = getattr(rl, attr, None)
        if val is None:
            # Try CamelCase fallback
            parts = attr.split('_')
            camel = parts[0] + ''.join(x.title() for x in parts[1:])
            val = getattr(rl, camel, None)
        
        if val is not None:
            try:
                pos = getattr(val, "position", 0)
                dtype = getattr(val, "datatype", getattr(val, "data_type", "UBYTE"))
                components.append(LayoutComponent(attr, pos, dtype, relation))
            except Exception:
                pass

    # 2. Check 'axes' dictionary (nested structure)
    axes_attr = getattr(rl, "axes", None)
    if axes_attr:
        for axis_key, axis_data in axes_attr.items():
            # axis_key is 'x', 'y', 'z', etc.
            if not isinstance(axis_data, dict):
                continue
                
            # Check for axis_pts
            if "axis_pts" in axis_data:
                val = axis_data["axis_pts"]
                try:
                    pos = getattr(val, "position", 0)
                    dtype = getattr(val, "datatype", getattr(val, "data_type", "UBYTE"))
                    name = f"axis_pts_{axis_key}"
                    components.append(LayoutComponent(name, pos, dtype, "AXIS_PTS"))
                except Exception:
                    pass

            # Check for no_axis_pts
            if "no_axis_pts" in axis_data:
                val = axis_data["no_axis_pts"]
                try:
                    pos = getattr(val, "position", 0)
                    dtype = getattr(val, "datatype", getattr(val, "data_type", "UBYTE"))
                    name = f"no_axis_pts_{axis_key}"
                    components.append(LayoutComponent(name, pos, dtype, "NO_AXIS"))
                except Exception:
                    pass

    # Deduplicate components by position (in case we found same component via both methods)
    # Prefer the one with a valid name matching our attrs map if possible, or just keep one.
    unique_components = {}
    for c in components:
        unique_components[c.position] = c
    components = list(unique_components.values())

    # Sort by position
    components.sort(key=lambda x: x.position)
    logging.debug("Parsed Record Layout %s: %s", getattr(rl, 'name', 'unknown'), components)
    return components


def parse_address_mappings(a2l_path: str) -> List[Dict[str, int]]:
    """
    Parse ADDRESS_MAPPING IF_DATA blocks directly from the A2L text.
    Falls back to empty list if not found or on error.
    """
    mappings = []
    try:
        text = pathlib.Path(a2l_path).read_text(errors="ignore")
    except Exception as ex:
        logging.debug("Could not read A2L for ADDRESS_MAPPING: %s", ex)
        return mappings

    import re
    # Example line: ADDRESS_MAPPING /*orig_adr:*/0x1C2000 /*mapping_adr:*/0x5C2000 /*length:*/0x1E000
    pattern = re.compile(
        r"ADDRESS_MAPPING\s*/\*orig_adr:\*/0x([0-9A-Fa-f]+)\s*/\*mapping_adr:\*/0x([0-9A-Fa-f]+)\s*/\*length:\*/0x([0-9A-Fa-f]+)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        try:
            orig = int(m.group(1), 16)
            mapped = int(m.group(2), 16)
            length = int(m.group(3), 16)
            mappings.append({"orig": orig, "mapped": mapped, "length": length})
        except Exception as ex:
            logging.debug("Failed to parse mapping entry: %s", ex)

    if mappings:
        logging.info("Parsed %d ADDRESS_MAPPING entries", len(mappings))
    return mappings


def map_address(addr: int) -> int:
    """
    Apply ADDRESS_MAPPING remap if the address falls within a mapped region.
    """
    for m in address_mappings:
        if m["orig"] <= addr < m["orig"] + m["length"]:
            return m["mapped"] + (addr - m["orig"])
    return addr


def region_overlaps(ranges):
    """Given list of (start, end), return count of overlaps."""
    overlaps = 0
    ranges = sorted(ranges, key=lambda x: x[0])
    prev_end = None
    for start, end in ranges:
        if prev_end is not None and start <= prev_end:
            overlaps += 1
        prev_end = max(prev_end, end) if prev_end is not None else end
    return overlaps

# Statistics counters
axis_stats = Counter()
char_type_stats = Counter()  # Track characteristic types
measurement_stats = Counter()  # Track measurement data types

# Mapping from A2L data types to XDF/ADX data sizes
# Based on ASAM MCD-2 MC specification v1.6/v1.7
data_sizes: Dict[str, int] = {
    "UBYTE": 1,        # Unsigned 8-bit
    "SBYTE": 1,        # Signed 8-bit
    "UWORD": 2,        # Unsigned 16-bit
    "SWORD": 2,        # Signed 16-bit
    "ULONG": 4,        # Unsigned 32-bit
    "SLONG": 4,        # Signed 32-bit
    "A_UINT64": 8,     # Unsigned 64-bit
    "A_INT64": 8,      # Signed 64-bit
    "FLOAT16_IEEE": 2, # IEEE 754 half precision (rare)
    "FLOAT32_IEEE": 4, # IEEE 754 single precision
    "FLOAT64_IEEE": 8, # IEEE 754 double precision
}


def get_data_size(datatype: str) -> int:
    """Get size for data type with fallback for unknown types."""
    if not datatype:
        logging.warning("Empty data type, defaulting to 1 byte")
        return 1
    if datatype not in data_sizes:
        logging.warning(f"Unknown data type '{datatype}', defaulting to 4 bytes")
        return 4
    return data_sizes[datatype]


def xdf_root_with_configuration(title: str, region_size: int = DEFAULT_REGION_SIZE):
    root = Element("XDFFORMAT")
    root.set("version", "1.60")

    xdfheader = SubElement(root, "XDFHEADER")

    flags = SubElement(xdfheader, "flags")
    flags.text = "0x1"

    deftitle = SubElement(xdfheader, "deftitle")
    deftitle.text = title

    description = SubElement(xdfheader, "description")
    description.text = "Auto-generated by A2L2XDF (no CSV)"

    baseoffset = SubElement(xdfheader, "BASEOFFSET")
    baseoffset.set("offset", "0")
    baseoffset.set("subtract", "0")

    defaults = SubElement(xdfheader, "DEFAULTS")
    defaults.set("datasizeinbits", "8")
    defaults.set("sigdigits", "4")
    defaults.set("outputtype", "1")
    defaults.set("signed", "0")
    defaults.set("lsbfirst", "1")
    defaults.set("float", "0")

    region = SubElement(xdfheader, "REGION")
    region.set("type", "0xFFFFFFFF")
    region.set("startaddress", "0x0")
    region.set("size", hex(region_size))
    region.set("regionflags", "0x0")
    region.set("name", "Binary")
    region.set("desc", "BIN region")

    return root, xdfheader


def new_unique_id():
    return hex(uuid.uuid4().int & 0xFFFFFFFF)


def xdf_category(xdfheader, name, idx):
    c = SubElement(xdfheader, "CATEGORY")
    c.set("index", str(idx))
    c.set("name", name)
    return c


def add_category(xdfheader, name):
    if name not in categories:
        categories.append(name)
        xdf_category(xdfheader, name, len(categories))


def add_table_categories(table, cats):
    for i, c in enumerate(cats):
        cm = SubElement(table, "CATEGORYMEM")
        cm.set("index", str(i))
        cm.set("category", str(categories.index(c) + 1))


def fix_degree(v: Optional[str]) -> str:
    """Replace Unicode replacement character with degree symbol. Handle None values."""
    if v is None:
        return ""
    return re.sub("\uFFFD", "\u00B0", str(v))


def adjust_address(addr: int, base: int = None) -> int:
    """
    Convert ECU ROM address to file offset for XDF/BIN access using a single base.
    ADDRESS_MAPPING is not applied here; XDF should point at raw file offsets.
    """
    if base is None:
        base = BASE_OFFSET

    return addr - base


def adjust_ram_address(addr: int) -> int:
    """
    RAM/ECU live addresses for ADX: apply ADDRESS_MAPPING if present,
    but do NOT subtract ROM base (we want ECU addresses for logging).
    """
    return map_address(addr)


def addr_range(start: int, elem_bits: int, count: int, alignment_bits: int = 0, stride_bits: Optional[int] = None):
    """
    Compute end address of a data block considering alignment and optional stride.
    alignment_bits applies to the start of the block.
    """
    if alignment_bits:
        aligned_start_bits = ((start * 8 + alignment_bits - 1) // alignment_bits) * alignment_bits
        start = aligned_start_bits // 8
    stride = stride_bits if stride_bits is not None else elem_bits
    end = start + ((count - 1) * stride) // 8 + (elem_bits // 8)
    return start, end


def coefficients_to_equation(coeffs) -> str:
    """
    Convert ASAM RAT_FUNC coefficient object to equation string.

    ASAM MCD-2 MC RAT_FUNC formula:
        Physical = (a*X² + b*X + c) / (d*X² + e*X + f)

    Common simplifications:
        - LINEAR (a=0, d=0, e=0, f=1): Physical = (b*X + c)
        - OFFSET (a=0, b=1, d=0, e=0, f=1): Physical = X + c
        - SCALE (a=0, c=0, d=0, e=0, f=1): Physical = b*X
    """
    if coeffs is None:
        return "X"

    try:
        # Try both dict-like and attribute-like access
        def get(name, default=0.0):
            if hasattr(coeffs, name):
                return float(getattr(coeffs, name))
            if hasattr(coeffs, "get"):
                return float(coeffs.get(name, default))
            return float(default)

        a = get("a")
        b = get("b")
        c = get("c")
        d = get("d")
        e = get("e")
        f_ = get("f")
    except (AttributeError, KeyError, TypeError, ValueError) as ex:
        logging.debug(f"Failed to parse coefficients: {ex}")
        return "X"

    # Check for division by zero risk
    if f_ == 0.0 and e == 0.0 and d == 0.0:
        logging.warning("RAT_FUNC denominator is zero, using identity")
        return "X"

    # Build numerator: a*X² + b*X + c
    numerator_terms = []
    if a != 0.0:
        numerator_terms.append(f"({a} * X * X)")
    if b != 0.0:
        numerator_terms.append(f"({b} * X)")
    if c != 0.0:
        numerator_terms.append(f"{c}")

    if not numerator_terms:
        numerator = "0.0"
    else:
        numerator = " + ".join(numerator_terms)

    # Build denominator: d*X² + e*X + f
    denominator_terms = []
    if d != 0.0:
        denominator_terms.append(f"({d} * X * X)")
    if e != 0.0:
        denominator_terms.append(f"({e} * X)")
    if f_ != 0.0:
        denominator_terms.append(f"{f_}")

    if not denominator_terms:
        denominator = "1.0"
    else:
        denominator = " + ".join(denominator_terms)

    # Special case: Linear (no quadratic terms, denominator = 1)
    if a == 0.0 and d == 0.0 and e == 0.0 and f_ == 1.0:
        if b == 1.0 and c == 0.0:
            return "X"  # Identity
        elif b == 1.0:
            return f"X + {c}"  # Offset
        elif c == 0.0:
            return f"{b} * X"  # Scale
        else:
            return f"({b} * X) + {c}"  # Linear

    # General rational function
    return f"({numerator}) / ({denominator})"


def resolve_compu_method(session, compu_name: Optional[str]) -> Tuple[str, str]:
    """Return (math, units) for a COMPU_METHOD name."""
    if not compu_name:
        return "X", ""

    try:
        compu = session.query(model.CompuMethod).filter(model.CompuMethod.name == compu_name).first()
    except Exception as ex:
        logging.debug("Failed to resolve compu method %s: %s", compu_name, ex)
        return "X", ""

    if compu is None:
        return "X", ""

    units = fix_degree(getattr(compu, 'unit', '') or '')

    conversion = getattr(compu, 'conversionType', None) or getattr(compu, 'conversion_type', None)
    coeffs = getattr(compu, 'coeffs', None)

    # TAB_INTP with exactly two points -> derive linear equation
    if conversion and "TAB_INTP" in str(conversion).upper():
        eq = compu_interp_equation(session, compu_name)
        if eq:
            return eq, units

    if coeffs is not None:
        return coefficients_to_equation(coeffs), units

    if conversion and "IDENT" in str(conversion).upper():
        return "X", units

    return "X", units


def compu_enum_labels(session, compu_name: Optional[str]) -> Dict[str, str]:
    """Return a mapping for TAB_VERB/TAB_INTP discrete labels, if present."""
    if not compu_name:
        return {}
    try:
        compu = session.query(model.CompuMethod).filter(model.CompuMethod.name == compu_name).first()
    except Exception:
        return {}

    if compu is None:
        return {}

    table = getattr(compu, 'verbalTable', None) or getattr(compu, 'verbal_table', None)
    pairs = getattr(compu, 'valuePairs', None) or getattr(compu, 'value_pairs', None)

    mapping: Dict[str, str] = {}
    for src in (table, pairs):
        if not src:
            continue
        try:
            for entry in src:
                key = str(getattr(entry, 'value', getattr(entry, 'inVal', None)))
                label = str(getattr(entry, 'text', getattr(entry, 'outVal', None)))
                if key is not None and label is not None:
                    mapping[key] = label
        except Exception:
            continue

    return mapping


def compu_bit_enums(bitmask: int, labels: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Build enumerations for bitmask:
      - single-bit: Off/On unless labels provided.
      - small multi-bit (<=4 bits wide): enumerate all values with generic labels.
    """
    if bitmask is None:
        return {}
    # single-bit
    if bitmask & (bitmask - 1) == 0:
        return labels or {"0": "Off", "1": "On"}

    # small multi-bit (width <=4)
    width = bitmask.bit_length()
    if width <= 4:
        enum = {}
        for v in range(0, 1 << width):
            enum[str(v)] = labels.get(str(v), f"Val{v}") if labels else f"Val{v}"
        return enum

    return labels or {}


def compu_range_labels(session, compu_name: Optional[str]) -> Dict[str, str]:
    """Return a mapping for COMPU_VTAB_RANGE as 'low-high': label strings."""
    mapping: Dict[str, str] = {}
    if not compu_name:
        return mapping
    try:
        compu = session.query(model.CompuMethod).filter(model.CompuMethod.name == compu_name).first()
    except Exception:
        return mapping
    if compu is None:
        return mapping

    ranges = getattr(compu, "verbalTableRange", None) or getattr(compu, "verbal_table_range", None)
    if not ranges:
        return mapping
    try:
        for entry in ranges:
            lo = getattr(entry, "lowerLimit", None)
            hi = getattr(entry, "upperLimit", None)
            text = getattr(entry, "text", None)
            if lo is None or hi is None or text is None:
                continue
            mapping[f"{lo}-{hi}"] = str(text)
    except Exception:
        return mapping
    return mapping


def extract_bitmask(obj) -> Optional[int]:
    """Return integer bitmask if present on object."""
    bm = getattr(obj, "bitMask", None) or getattr(obj, "bit_mask", None)
    if bm is None:
        return None
    try:
        if hasattr(bm, "mask"):
            return int(bm.mask)
        if hasattr(bm, "value"):
            return int(bm.value)
        return int(bm)
    except Exception:
        return None


def compu_interp_equation(session, compu_name: Optional[str]) -> Optional[str]:
    """
    For TAB_INTP/VTAB_INTP with exactly two points, derive a linear equation TunerPro can use.
    Returns equation string or None if not derivable.
    """
    if not compu_name:
        return None
    try:
        compu = session.query(model.CompuMethod).filter(model.CompuMethod.name == compu_name).first()
    except Exception:
        return None
    if compu is None:
        return None

    tab = getattr(compu, "compuTabRef", None) or getattr(compu, "compu_tab_ref", None)
    if tab is None:
        return None
    try:
        table = session.query(model.CompuTab).filter(model.CompuTab.name == tab).first()
    except Exception:
        return None
    if table is None:
        return None

    pairs = getattr(table, "valuePairs", None) or getattr(table, "value_pairs", None)
    if not pairs or len(pairs) < 2:
        return None

    try:
        x1, y1 = float(pairs[0].inVal), float(pairs[0].outVal)
        x2, y2 = float(pairs[-1].inVal), float(pairs[-1].outVal)
    except Exception:
        return None

    if x2 == x1:
        return None

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return f"({slope} * X) + ({intercept})"


def compu_piecewise_labels(session, compu_name: Optional[str]) -> Dict[str, str]:
    """Create coarse range labels for TAB_INTP/VTAB_INTP with >2 points."""
    labels: Dict[str, str] = {}
    if not compu_name:
        return labels
    try:
        compu = session.query(model.CompuMethod).filter(model.CompuMethod.name == compu_name).first()
    except Exception:
        return labels
    if compu is None:
        return labels
    tab = getattr(compu, "compuTabRef", None) or getattr(compu, "compu_tab_ref", None)
    if tab is None:
        return labels
    try:
        table = session.query(model.CompuTab).filter(model.CompuTab.name == tab).first()
    except Exception:
        return labels
    if table is None:
        return labels
    pairs = getattr(table, "valuePairs", None) or getattr(table, "value_pairs", None)
    if not pairs or len(pairs) < 3:
        return labels
    try:
        for i in range(len(pairs) - 1):
            x1, y1 = float(pairs[i].inVal), float(pairs[i].outVal)
            x2, y2 = float(pairs[i+1].inVal), float(pairs[i+1].outVal)
            mid_y = (y1 + y2) / 2.0
            labels[f"{x1}-{x2}"] = f"~{mid_y}"
    except Exception:
        return labels
    return labels


def get_axis_layout_from_deposit(deposit_attr):
    """
    Extract the X-axis 'axis_pts' layout from depositAttr in a version-tolerant way.

    pya2l can expose depositAttr.axes in a few shapes:
      - dict-like:   axes["x"]["axis_pts"]
      - object-like: axes.x.axis_pts

    Returns the axis layout object or None.
    """
    if deposit_attr is None:
        return None

    axes_obj = getattr(deposit_attr, "axes", None)
    if axes_obj is None:
        return None

    # Get X axis object
    if isinstance(axes_obj, dict):
        x_axis = axes_obj.get("x")
    else:
        x_axis = getattr(axes_obj, "x", None)

    if x_axis is None:
        return None

    # Get axis_pts layout
    if isinstance(x_axis, dict):
        axis_layout = x_axis.get("axis_pts")
    else:
        axis_layout = getattr(x_axis, "axis_pts", None)

    return axis_layout


def resolve_axis_source(ar: inspect.AxisDescr):
    """
    Return the underlying axis object for this AxisDescr and a short tag for logging.

    Priority:
      1. axisPtsRef (AXIS_PTS)
      2. comAxisRef (COM_AXIS, if present)
      3. axisDescrRef / axisRef (other link types, if present)
      4. The AxisDescr itself (STD_AXIS with own depositAttr)
    """
    axis = getattr(ar, "axisPtsRef", None)
    if axis is not None:
        return axis, "AXIS_PTS_REF"

    com_axis = getattr(ar, "comAxisRef", None)
    if com_axis is not None:
        return com_axis, "COM_AXIS_REF"

    axis_descr_ref = getattr(ar, "axisDescrRef", None)
    if axis_descr_ref is not None:
        return axis_descr_ref, "AXIS_DESCR_REF"

    axis_ref = getattr(ar, "axisRef", None)
    if axis_ref is not None:
        return axis_ref, "AXIS_REF"

    # Fallback: use the AxisDescr itself – covers STD_AXIS + DEPOSIT ABSOLUTE
    return ar, "AXIS_DESCR_SELF"


def generate_fixed_axis_values(ar, axis_type: str):
    """
    Build a list of *physical* axis values for FIX_AXIS* style axes.

    Many Bosch A2Ls (like your ME9.6) declare FIX_AXIS_PAR / FIX_AXIS_PAR_DIST
    but leave the parameters None. In that case we fall back to a simple,
    evenly spaced axis between lowerLimit and upperLimit.

    Returns:
        list[float] or None if we can't sensibly generate an axis.
    """

    # 1) Work out how many points the axis should have
    n = None

    # Try FIX_AXIS_PAR / FIX_AXIS_PAR_DIST.numberapo first, if present
    fa = getattr(ar, "fixAxisPar", None)
    fad = getattr(ar, "fixAxisParDist", None)

    for src in (fa, fad):
        if src is not None:
            na = getattr(src, "numberapo", None)
            if na is not None:
                n = na
                break

    # Fall back to maxAxisPoints from the AXIS_DESCR itself
    if n is None:
        n = getattr(ar, "maxAxisPoints", None)

    if not n:
        logging.debug(
            "Axis %r (%s): no valid number of points; skipping fixed-axis generation",
            getattr(ar, "inputQuantity", "<??>"),
            axis_type,
        )
        return None

    n = int(n)

    # 2) Get the physical range
    lo = getattr(ar, "lowerLimit", 0.0)
    hi = getattr(ar, "upperLimit", lo)

    # Guard against None
    lo = float(lo if lo is not None else 0.0)
    hi = float(hi if hi is not None else lo)

    # 3) Degenerate cases
    if n <= 0:
        logging.debug(
            "Axis %r (%s): non-positive point count (%d); skipping",
            getattr(ar, "inputQuantity", "<??>"),
            axis_type,
            n,
        )
        return None

    if n == 1:
        logging.debug(
            "Axis %r (%s): single-point fixed axis at %f",
            getattr(ar, "inputQuantity", "<??>"),
            axis_type,
            lo,
        )
        return [lo]

    # 4) Normal case: evenly spaced between lowerLimit and upperLimit
    step = (hi - lo) / (n - 1) if n > 1 else 0.0
    values = [lo + i * step for i in range(n)]

    logging.debug(
        "Axis %r (%s): generated %d fixed axis values from [%f, %f] (step=%f)",
        getattr(ar, "inputQuantity", "<??>"),
        axis_type,
        n,
        lo,
        hi,
        step,
    )

    return values


def detect_axis_type(ar: inspect.AxisDescr) -> str:
    """Detect what type of axis definition this is."""

    # Curve axis reference (used sometimes to share axes)
    if getattr(ar, "curveAxisRef", None) is not None:
        return "CURVE_AXIS_REF"

    # General axis source: AXIS_PTS, COM_AXIS, AXIS_DESCR itself, etc.
    axis_obj, source_tag = resolve_axis_source(ar)
    if axis_obj is None:
        logging.debug(
            "AxisDescr %r has no usable axis source (axisPtsRef / comAxisRef / axisDescrRef / axisRef)",
            ar,
        )
        return "NO_AXIS_SOURCE"

    # Memory-mapped layout via depositAttr covers:
    #   - COM_AXIS with AXIS_PTS_REF
    #   - STD_AXIS + DEPOSIT ABSOLUTE (axis points stored with the curve)
    deposit_attr = getattr(axis_obj, "depositAttr", None)
    axis_layout = get_axis_layout_from_deposit(deposit_attr)
    if axis_layout is not None:
        return "MEMORY_MAPPED"

    # If we got here, see if it's at least an ABSOLUTE-deposit axis with no layout
    deposit = getattr(axis_obj, "deposit", None)
    if deposit == "ABSOLUTE":
        # Check if this is a STD_AXIS - we can still use it with maxAxisPoints
        axis_type_attr = getattr(ar, "attribute", None) or getattr(ar, "type", None)
        if axis_type_attr and "STD_AXIS" in str(axis_type_attr):
            return "STD_AXIS_DEPOSIT_ABSOLUTE"
        return "DEPOSIT_ABSOLUTE_NO_PARAMS"

    # Fixed axes (FIX_AXIS) - Check LAST as fallback if no memory layout found
    if getattr(ar, "fixAxisPar", None) is not None:
        return "FIX_AXIS_PAR"

    if getattr(ar, "fixAxisParDist", None) is not None:
        return "FIX_AXIS_PAR_DIST"

    if getattr(ar, "fixAxisParList", None) is not None:
        return "FIX_AXIS_PAR_LIST"

    logging.debug(
        "AxisDescr %r with source %s has no axis layout and no known deposit type",
        ar, source_tag,
    )
    return "UNKNOWN_TYPE"


def _axis_ref_to_dict_inner(ar: inspect.AxisDescr, depth: int = 0) -> Optional[Dict[str, Any]]:
    """
    Internal worker for axis_ref_to_dict with recursion depth tracking
    (used for CURVE_AXIS_REF).
    """
    if depth > MAX_AXIS_RESOLUTION_DEPTH:
        logging.debug("Axis resolution depth exceeded for %r, aborting", ar)
        return None

    axis_type = detect_axis_type(ar)
    axis_stats[axis_type] += 1

    axis_obj, source_tag = resolve_axis_source(ar)
    axis_name = "unknown"
    axis_description = None
    input_quantity = None
    units = ""
    compu = None
    enum_map: Dict[str, str] = {}
    range_map: Dict[str, str] = {}

    if axis_obj is not None:
        # Get the basic name
        axis_name = getattr(axis_obj, "name", axis_name)
        
        # Get longIdentifier (detailed description) if available
        axis_description = getattr(axis_obj, "longIdentifier", None)
        
        # Get inputQuantity (physical meaning) if available
        input_quantity = getattr(axis_obj, "inputQuantity", None)
        
        # Get COMPU_METHOD for units
        compu = getattr(axis_obj, "compuMethod", None)

    # Fallback: try to get inputQuantity and compu from AXIS_DESCR itself
    if input_quantity is None:
        input_quantity = getattr(ar, "inputQuantity", None)
    
    if compu is None:
        compu = getattr(ar, "compuMethod", None)

    # Build a rich, user-friendly axis name
    # Format: "inputQuantity (axis_name): description"
    # Example: "nmot (MDV08_UC): Datapoint distribution for torque in instruction test"
    display_name = axis_name
    if input_quantity and input_quantity != axis_name:
        display_name = f"{input_quantity} ({axis_name})"
    
    if axis_description:
        display_name = f"{display_name}: {axis_description}"

    if compu is not None:
        units = fix_degree(compu.unit)
        enum_map = compu_enum_labels_from_obj(compu)
        range_map = compu_range_labels_from_obj(compu)
        range_map = compu_range_labels_from_obj(compu)

    # MEMORY-MAPPED AXES (STD_AXIS, COM_AXIS, AXIS_PTS, etc.)
    if axis_type == "MEMORY_MAPPED":
        deposit_attr = getattr(axis_obj, "depositAttr", None)
        axis_layout = get_axis_layout_from_deposit(deposit_attr)
        if axis_layout is None:
            logging.debug(
                "Axis %r detected as MEMORY_MAPPED (source=%s) but no axis_layout found",
                axis_name, source_tag,
            )
            return None

        datatype = getattr(axis_layout, "data_type", None) or getattr(axis_obj, "datatype", None) or "UBYTE"
        addr_raw = getattr(axis_layout, "address", None)
        if addr_raw is None:
            logging.debug("Axis %r layout has no 'address' attribute", axis_name)
            return None

        try:
            addr_int = int(addr_raw)
        except Exception:
            logging.debug("Axis %r address not int-convertible (%r)", axis_name, addr_raw)
            return None

        addr = adjust_address(addr_int)

        coeffs = compu.coeffs if (compu is not None and hasattr(compu, "coeffs")) else None
        math = coefficients_to_equation(coeffs) if coeffs is not None else "X"

        length = getattr(ar, "maxAxisPoints", None) or getattr(ar, "max_axis_points", None)
        if length is None:
            length = getattr(axis_layout, "num_points", 0) or 1
        stride_bytes = get_dist_op_stride(axis_layout)

        return {
            "name": display_name,
            "units": units,
            "min": getattr(ar, "lowerLimit", 0.0),
            "max": getattr(ar, "upperLimit", 0.0),
            "address": hex(addr),
            "length": int(length),
            "dataSize": datatype,
            "math": math,
            "lsb_first": is_little_endian(axis_layout),
            "enum": enum_map,
            "enum_range": range_map,
            "stride_bytes": stride_bytes,
        }

    # FIXED / COMPUTED AXES
    if axis_type in ("FIX_AXIS_PAR", "FIX_AXIS_PAR_DIST", "FIX_AXIS_PAR_LIST"):
        values = generate_fixed_axis_values(ar, axis_type)
        if not values:
            logging.debug(
                "Axis %r has axis_type=%s but generated no values; skipping",
                axis_name, axis_type,
            )
            return None

        return {
            "name": display_name,
            "units": units,
            "min": min(values),
            "max": max(values),
            "address": None,          # no memory address – computed axis
            "length": len(values),
            "dataSize": None,
            "math": "X",
            "values": values,         # used by real_axis() for LABELs
            "lsb_first": True,
            "enum": enum_map,
            "enum_range": range_map,
        }

    # CURVE-BASED AXES
    if axis_type == "CURVE_AXIS_REF":
        curve = getattr(ar, "curveAxisRef", None)
        if curve is None:
            logging.debug("Axis %r has CURVE_AXIS_REF but no curveAxisRef object", axis_name)
            return None

        axis_list = getattr(curve, "axisDescriptions", None)
        if not axis_list:
            logging.debug(
                "Axis %r CURVE_AXIS_REF but referenced curve has no axisDescriptions",
                axis_name,
            )
            return None

        logging.debug(
            "Resolving CURVE_AXIS_REF for axis %r via curve %r",
            axis_name, getattr(curve, "name", "unknown"),
        )
        ref_axis_descr = axis_list[0]
        return _axis_ref_to_dict_inner(ref_axis_descr, depth + 1)

    # STD_AXIS with DEPOSIT ABSOLUTE but no detailed layout info
    # The axis is stored inline - we need to mark it for special handling
    # Since we don't have the characteristic address here, we return a marker
    # that tells the caller to handle this specially
    if axis_type == "STD_AXIS_DEPOSIT_ABSOLUTE":
        length = getattr(ar, "maxAxisPoints", None)
        if length is None:
            logging.debug(
                "Axis %r has STD_AXIS_DEPOSIT_ABSOLUTE but no maxAxisPoints",
                axis_name,
            )
            return None

        coeffs = compu.coeffs if (compu is not None and hasattr(compu, "coeffs")) else None
        math = coefficients_to_equation(coeffs) if coeffs is not None else "X"

        # Return with special marker - address will be calculated by caller
        return {
            "name": display_name,
            "units": units,
            "min": getattr(ar, "lowerLimit", 0.0),
            "max": getattr(ar, "upperLimit", 0.0),
            "address": None,  # Will be set by caller based on characteristic layout
            "length": int(length),
            "dataSize": "UBYTE",  # Default, may need to be determined from record layout
            "math": math,
            "inline_axis": True,  # Marker for inline axis handling
            "lsb_first": True,
            "enum": enum_map,
            "enum_range": range_map,
        }

    logging.debug(
        "Skipping axis %r - unsupported axis_type=%s (source=%s)",
        axis_name, axis_type, source_tag,
    )
    return None


def axis_ref_to_dict(ar: inspect.AxisDescr) -> Optional[Dict[str, Any]]:
    """
    Public wrapper: convert axis reference to dictionary. Returns None on error.

    Handles:
      - MEMORY_MAPPED (STD_AXIS / COM_AXIS / AXIS_PTS)
      - FIX_AXIS_PAR / FIX_AXIS_PAR_DIST / FIX_AXIS_PAR_LIST
      - CURVE_AXIS_REF by following the referenced curve.
      - STD_AXIS_DEPOSIT_ABSOLUTE (inline axis, address calculated by caller)
    """
    return _axis_ref_to_dict_inner(ar, depth=0)


def xdf_embedded(table, axis_id, axis_def):
    """
    Emit EMBEDDEDDATA with correct stride semantics for TunerPro.

    Handles both ROW_DIR (row-major) and COLUMN_DIR (column-major) storage.

    For ROW_DIR (default):
        - Data stored row by row: Z[0,0], Z[0,1], ..., Z[0,cols-1], Z[1,0], ...
        - Major stride = elem_bits * cols (one complete row)
        - Minor stride = elem_bits (next column in same row)

    For COLUMN_DIR:
        - Data stored column by column: Z[0,0], Z[1,0], ..., Z[rows-1,0], Z[0,1], ...
        - Major stride = elem_bits * rows (one complete column)
        - Minor stride = elem_bits (next row in same column)
        - Need to SWAP rows/cols in XDF since TunerPro expects row-major
    """
    e = SubElement(table, "EMBEDDEDDATA")
    flags = XDF_FLAG_AXIS if axis_id != "z" else (XDF_FLAG_AXIS | XDF_FLAG_TABLE)

    data_size = axis_def.get("dataSize")
    elem_bits = get_data_size(data_size) * 8 if data_size else 8
    lsb_first = axis_def.get("lsb_first", True)

    # Handle floating point types (all IEEE floats are implicitly signed)
    if data_size in ("FLOAT16_IEEE", "FLOAT32_IEEE", "FLOAT64_IEEE"):
        flags |= XDF_FLAG_FLOAT   # Float flag
        flags |= XDF_FLAG_SIGNED  # Signed flag (IEEE floats are signed)

    # Handle signed integer types
    elif data_size in ("SBYTE", "SWORD", "SLONG", "A_INT64"):
        flags |= XDF_FLAG_SIGNED  # Signed flag

    e.set("mmedtypeflags", hex(flags))
    e.set("mmedaddress", str(axis_def.get("address", "0x0")))
    e.set("mmedlsbfirst", "1" if lsb_first else "0")

    e.set("mmedelementsizebits", str(elem_bits))

    # Get index mode (ROW_DIR or COLUMN_DIR)
    index_mode = axis_def.get("index_mode", "ROW_DIR")
    cols = int(axis_def.get("length", 1))
    rows = int(axis_def.get("rows", 1)) if axis_id == "z" else 1

    # For Z (tables), calculate stride based on storage order
    stride_bytes = axis_def.get("stride_bytes")
    if axis_id == "z" and rows > 1:
        if index_mode == "COLUMN_DIR":
            # Column-major: major stride is one column (rows elements)
            # TunerPro expects row-major, so we need to SWAP rows/cols
            major = (stride_bytes * 8) if stride_bytes else (elem_bits * rows)
            e.set("mmedmajorstridebits", str(major))
            e.set("mmedminorstridebits", str(elem_bits))
            # Swap rows and cols for column-major data
            e.set("mmedcolcount", str(rows))  # Swapped!
            e.set("mmedrowcount", str(cols))  # Swapped!
        else:
            # Row-major (default): major stride is one row (cols elements)
            major = (stride_bytes * 8) if stride_bytes else (elem_bits * cols)
            e.set("mmedmajorstridebits", str(major))
            e.set("mmedminorstridebits", str(elem_bits))
            e.set("mmedcolcount", str(cols))
            e.set("mmedrowcount", str(rows))
    else:
        # For axes (1D) or single values
        e.set("mmedmajorstridebits", str(elem_bits))
        e.set("mmedminorstridebits", "0")
        e.set("mmedcolcount", str(cols))
        if axis_id == "z":
            e.set("mmedrowcount", str(rows))

    return e


def fake_axis(table, axis_id, size):
    a = SubElement(table, "XDFAXIS")
    a.set("uniqueid", new_unique_id())
    a.set("id", axis_id)

    SubElement(a, "indexcount").text = str(size)
    SubElement(a, "outputtype").text = "4"

    d = SubElement(a, "DALINK")
    d.set("index", "0")

    m = SubElement(a, "MATH")
    m.set("equation", "X")
    v = SubElement(m, "VAR")
    v.set("id", "X")

    for i in range(size):
        lab = SubElement(a, "LABEL")
        lab.set("index", str(i))
        lab.set("value", "-")


def real_axis(table: Element, axis_id: str, axis_def: Dict[str, Any]) -> None:
    """Create a real axis element with all required XDF sub-elements."""
    a = SubElement(table, "XDFAXIS")
    a.set("uniqueid", new_unique_id())
    a.set("id", axis_id)

    length = int(axis_def.get("length", 1))
    min_val = axis_def.get("min", 0.0)
    max_val = axis_def.get("max", 0.0)
    units = axis_def.get("units", "")
    math = axis_def.get("math", "X")
    address = axis_def.get("address")
    data_size = axis_def.get("dataSize")
    values: Optional[List[Any]] = axis_def.get("values")

    # Only build EMBEDDEDDATA when we have a memory-backed axis
    if address is not None and data_size is not None:
        embedded_def = {
            "address": address,
            "dataSize": data_size,
            "length": length,
            "lsb_first": axis_def.get("lsb_first", True),
        }
        if axis_id == "z" and "rows" in axis_def:
            embedded_def["rows"] = axis_def["rows"]

        embedded_def = {k: v for k, v in embedded_def.items() if v is not None}
        xdf_embedded(a, axis_id, embedded_def)

    SubElement(a, "indexcount").text = str(length)
    SubElement(a, "min").text = str(min_val)
    SubElement(a, "max").text = str(max_val)
    SubElement(a, "units").text = units

    # For fixed/computed axes, emit labels instead of relying on EMBEDDEDDATA
    if values:
        for i, val in enumerate(values):
            lab = SubElement(a, "LABEL")
            lab.set("index", str(i))
            lab.set("value", str(val))

    # Enumerations for axes (verbal tables)
    enum_map = axis_def.get("enum")
    enum_range = axis_def.get("enum_range")
    if enum_map or enum_range:
        add_enumeration(a, enum_map or {}, enum_range or {})

    # embedinfo element - required by XDF format
    ei = SubElement(a, "embedinfo")
    ei.set("type", "3")
    if address is not None:
        ei.set("linkobjid", str(address))

    # DALINK element - required by XDF format
    da = SubElement(a, "DALINK")
    da.set("index", "0")

    # MATH element - required by XDF format
    m = SubElement(a, "MATH")
    m.set("equation", math)
    v = SubElement(m, "VAR")
    v.set("id", "X")


def add_enumeration(table: Element, enum_map: Dict[str, str], range_map: Dict[str, str] = None) -> None:
    """Attach an ENUMERATION block for discrete value displays."""
    if not enum_map and not range_map:
        return
    en = SubElement(table, "ENUMERATION")
    for raw, text in (enum_map or {}).items():
        er = SubElement(en, "ENUM")
        er.set("value", str(raw))
        er.set("text", str(text))
    for rng, text in (range_map or {}).items():
        er = SubElement(en, "ENUM")
        er.set("value", rng)  # TunerPro will display text; range shown as string
        er.set("text", str(text))


def compu_enum_labels_from_obj(compu) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if compu is None:
        return mapping
    table = getattr(compu, 'verbalTable', None) or getattr(compu, 'verbal_table', None)
    pairs = getattr(compu, 'valuePairs', None) or getattr(compu, 'value_pairs', None)
    for src in (table, pairs):
        if not src:
            continue
        try:
            for entry in src:
                key = str(getattr(entry, 'value', getattr(entry, 'inVal', None)))
                label = str(getattr(entry, 'text', getattr(entry, 'outVal', None)))
                if key is not None and label is not None:
                    mapping[key] = label
        except Exception:
            continue
    return mapping


def compu_range_labels_from_obj(compu) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if compu is None:
        return mapping
    ranges = getattr(compu, "verbalTableRange", None) or getattr(compu, "verbal_table_range", None)
    if not ranges:
        return mapping
    try:
        for entry in ranges:
            lo = getattr(entry, "lowerLimit", None)
            hi = getattr(entry, "upperLimit", None)
            text = getattr(entry, "text", None)
            if lo is None or hi is None or text is None:
                continue
            mapping[f"{lo}-{hi}"] = str(text)
    except Exception:
        return mapping
    return mapping


def maybe_make_axis_table(root, table_def, name):
    addr = table_def[name].get("address")

    # Computed/fixed axes have no address: skip standalone axis table
    if addr is None:
        return

    if addr in axis_addresses_in_xdf:
        logging.debug("Axis address %s already emitted as standalone axis", addr)
        return
    axis_addresses_in_xdf.add(addr)

    t = SubElement(root, "XDFTABLE")
    t.set("uniqueid", new_unique_id())
    t.set("flags", "0x30")

    SubElement(t, "title").text = f"{table_def['title']} : {name} axis"
    SubElement(t, "description").text = table_def[name]["name"]

    add_table_categories(t, ["Axis"])

    fake_axis(t, "x", table_def[name]["length"])
    fake_axis(t, "y", 1)
    real_axis(t, "z", table_def[name])


def get_characteristic_type(ci, c) -> str:
    """Extract characteristic type string from inspect object or model object."""
    ctype = getattr(ci, "type", None)
    if ctype is None:
        ctype = getattr(c, "type", None)
    if ctype is None:
        return "UNKNOWN"
    return str(ctype)


def get_block_length(ci, c) -> int:
    """Get the NUMBER (block length) for VAL_BLK characteristics."""
    # Try inspect object first
    num = getattr(ci, "number", None)
    if num is None:
        # Fall back to model object
        num = getattr(c, "number", None)
    if num is None:
        return 1
    return int(num)


def infer_axis_datatype(ci, axis_idx: int) -> Optional[str]:
    """Best-effort lookup of axis point data type from record layout."""
    deposit = getattr(ci, "deposit", None)
    byte_order = is_little_endian(deposit) if deposit else True
    for attr in ("axisPts", "axis_pts", "axes"):
        axis_container = getattr(deposit, attr, None)
        if axis_container is None:
            continue

        # list/tuple form
        if isinstance(axis_container, (list, tuple)):
            if axis_idx < len(axis_container):
                dt = getattr(axis_container[axis_idx], "data_type", None)
                if dt:
                    return dt
        # dict-like form
        if isinstance(axis_container, dict):
            key = "x" if axis_idx == 0 else "y"
            maybe_axis = axis_container.get(key)
            if maybe_axis is not None:
                dt = getattr(maybe_axis, "data_type", None)
                if dt:
                    return dt

    return None


def write_pretty_xml(root, out):
    tree = ElementTree(root)
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    parsed = minidom.parseString(buf.getvalue())
    with open(out, "w", encoding="utf-8") as f:
        f.write(parsed.toprettyxml(indent="  "))
    print("  - Processed {} unique addresses".format(len(data_addresses_in_xdf)))
    print("  - Total characteristics: {}".format(len(char_type_stats) + sum(char_type_stats.values()) if not char_type_stats else sum(char_type_stats.values()))) # wait, char_type_stats has counts. sum() is correct.
    # Recalculate char stats sum correctly from Counter
    total_chars = sum(char_type_stats.values())
    print("  - Total characteristics: {}".format(total_chars))
    print("  - Total functions: {}".format(function_count))
    print("  - Functions with descriptions: {}".format(len(function_descriptions)))
    print("============================================================")


# =============================================================================
# ADX (TunerPro Data Acquisition) Generation for MEASUREMENTs
# =============================================================================

def adx_root_with_configuration(title: str, protocol: str = "CCP", pollrate: int = DEFAULT_ADX_POLL_RATE_MS):
    """Create ADX root element with header configuration."""
    root = Element("ADXFORMAT")
    root.set("version", "1.00")

    header = SubElement(root, "ADXHEADER")

    deftitle = SubElement(header, "deftitle")
    deftitle.text = title

    description = SubElement(header, "description")
    description.text = "Auto-generated by A2L2XDF - Data Acquisition Channels"

    # Default protocol settings (CCP/KWP/UDS)
    defaults = SubElement(header, "DEFAULTS")
    defaults.set("protocol", protocol)
    defaults.set("pollrate", str(pollrate))  # Default poll rate in ms

    return root, header


def adx_category(header, name, idx):
    """Add a category to the ADX header."""
    c = SubElement(header, "CATEGORY")
    c.set("index", str(idx))
    c.set("name", name)
    return c


def get_adx_datatype_flags(datatype: str) -> Dict[str, str]:
    """
    Convert A2L data type to ADX type flags.
    
    Returns dict with:
      - sizeinbits: Size in bits
      - signed: "1" if signed, "0" if unsigned
      - float: "1" if float, "0" otherwise
    """
    type_map = {
        "UBYTE":  {"sizeinbits": "8",  "signed": "0", "float": "0"},
        "SBYTE":  {"sizeinbits": "8",  "signed": "1", "float": "0"},
        "UWORD":  {"sizeinbits": "16", "signed": "0", "float": "0"},
        "SWORD":  {"sizeinbits": "16", "signed": "1", "float": "0"},
        "ULONG":  {"sizeinbits": "32", "signed": "0", "float": "0"},
        "SLONG":  {"sizeinbits": "32", "signed": "1", "float": "0"},
        "A_UINT64": {"sizeinbits": "64", "signed": "0", "float": "0"},
        "A_INT64":  {"sizeinbits": "64", "signed": "1", "float": "0"},
        "FLOAT16_IEEE": {"sizeinbits": "16", "signed": "1", "float": "1"},
        "FLOAT32_IEEE": {"sizeinbits": "32", "signed": "1", "float": "1"},
        "FLOAT64_IEEE": {"sizeinbits": "64", "signed": "1", "float": "1"},
    }
    
    if datatype not in type_map:
        logging.warning(f"Unknown ADX data type '{datatype}', defaulting to UBYTE")
        return type_map["UBYTE"]
    
    return type_map[datatype]


def create_adx_channel(root, meas_data: Dict[str, Any], idx: int, category_idx: int = 1, lsb_first: bool = True, enum_labels: Optional[Dict[str, str]] = None):
    """
    Create an ADX channel element for a MEASUREMENT.
    
    meas_data should contain:
      - name: Channel name
      - description: Description text
      - address: ECU address (hex string or int)
      - datatype: A2L data type (UBYTE, UWORD, etc.)
      - units: Unit string
      - min: Minimum value
      - max: Maximum value
      - math: Conversion equation
      - bitmask: Optional bit mask for boolean/flag values
      - array_size: Optional array size (default 1)
    """
    channel = SubElement(root, "ADXCHANNEL")
    channel.set("uniqueid", new_unique_id())
    
    # Title/name
    title = SubElement(channel, "title")
    title.text = meas_data.get("name", "Unknown")
    
    # Description
    desc = SubElement(channel, "description")
    desc.text = meas_data.get("description", "")
    
    # Units
    units = SubElement(channel, "units")
    units.text = meas_data.get("units", "")
    
    # Min/Max for display
    min_elem = SubElement(channel, "min")
    min_elem.text = str(meas_data.get("min", 0.0))
    
    max_elem = SubElement(channel, "max")
    max_elem.text = str(meas_data.get("max", 255.0))
    
    # Category membership
    catmem = SubElement(channel, "CATEGORYMEM")
    catmem.set("index", "0")
    catmem.set("category", str(category_idx))
    
    # Data address and type info
    address = meas_data.get("address", "0x0")
    if isinstance(address, int):
        address = hex(address)
    
    datatype = meas_data.get("datatype", "UBYTE")
    type_flags = get_adx_datatype_flags(datatype)
    
    # EMBEDDEDDATA element
    embed = SubElement(channel, "EMBEDDEDDATA")
    embed.set("mmedaddress", address)
    embed.set("mmedelementsizebits", type_flags["sizeinbits"])
    embed.set("mmedsigned", type_flags["signed"])
    embed.set("mmedfloat", type_flags["float"])
    embed.set("mmedlsbfirst", "1" if lsb_first else "0")

    # Optional per-channel pollrate hint
    if meas_data.get("pollrate") is not None:
        pr = SubElement(channel, "pollrate")
        pr.text = str(int(meas_data["pollrate"]))
    
    # Handle arrays
    array_size = meas_data.get("array_size", 1)
    if array_size > 1:
        embed.set("mmedcolcount", str(array_size))
    
    # Handle bit masks (for boolean/flag values)
    bitmask = meas_data.get("bitmask")
    if bitmask is not None:
        embed.set("mmedbitmask", hex(bitmask) if isinstance(bitmask, int) else str(bitmask))
    
    # MATH equation for conversion; if enums, prefer identity and store labels
    math_elem = SubElement(channel, "MATH")
    math_eq = meas_data.get("math", "X")
    math_elem.set("equation", math_eq)
    var = SubElement(math_elem, "VAR")
    var.set("id", "X")

    if enum_labels:
        for k, v in enum_labels.items():
            lab = SubElement(channel, "LABEL")
            lab.set("value", k)
            lab.set("text", v)

    # DAQ grouping placeholders (TunerPro expects payload); we at least set group id
    daq_group = meas_data.get("daq_group")
    if daq_group is not None:
        grp = SubElement(channel, "DAQGROUP")
        grp.set("id", str(daq_group))
        priority = meas_data.get("priority")
        if priority is not None:
            grp.set("priority", str(priority))

    # ADDRESS_MAPPING / variant info if present
    variant = meas_data.get("variant")
    if variant:
        var = SubElement(channel, "VARIANT")
        var.set("name", variant)
    
    return channel


def process_measurements(session, root, header, lsb_first: bool = True, default_pollrate: int = 100, args=None) -> int:
    """
    Process all MEASUREMENTs from the A2L and add them to the ADX.
    
    Returns the number of measurements processed.
    """
    measurements = session.query(model.Measurement).order_by(model.Measurement.name).all()
    logging.info("Found %d measurements", len(measurements))
    
    if not measurements:
        return 0
    
    # Create categories based on data type or prefix
    adx_categories: List[str] = []
    
    def add_adx_category(name: str) -> int:
        """Add category and return its 1-based index."""
        if name not in adx_categories:
            adx_categories.append(name)
            adx_category(header, name, len(adx_categories))
        return adx_categories.index(name) + 1
    
    # Default categories
    add_adx_category("Measurements")
    add_adx_category("Flags")
    add_adx_category("Arrays")
    
    processed_count = 0
    skipped = Counter()

    # Build EVENT lookup (priority, cycleTime)
    event_map = {}
    try:
        events = session.query(model.Event).all()
        for ev in events:
            ev_name = getattr(ev, "name", None)
            if ev_name:
                event_map[ev_name] = {
                    "cycle": getattr(ev, "cycleTime", None) or getattr(ev, "cycle_time", None),
                    "priority": getattr(ev, "priority", None),
                    "daq_id": getattr(ev, "daq_list", None) or getattr(ev, "daqId", None) or getattr(ev, "daq_id", None),
                }
    except Exception:
        pass
    
    for idx, m in enumerate(measurements):
        try:
            # Extract basic info
            name = m.name
            description = m.longIdentifier if m.longIdentifier else ""
            datatype = m.datatype if hasattr(m, 'datatype') else "UBYTE"
            
            # Get address from ECU_ADDRESS
            address = None
            if hasattr(m, 'ecuAddress') and m.ecuAddress is not None:
                ecu_addr = m.ecuAddress
                # Extract address value - may be an object with 'address' attribute
                if hasattr(ecu_addr, 'address'):
                    address = int(ecu_addr.address)
                else:
                    address = int(ecu_addr)
            elif hasattr(m, 'ecu_address') and m.ecu_address is not None:
                ecu_addr = m.ecu_address
                if hasattr(ecu_addr, 'address'):
                    address = int(ecu_addr.address)
                else:
                    address = int(ecu_addr)
            elif hasattr(m, 'address') and m.address is not None:
                address = int(m.address)

            if address is None:
                logging.debug("Skipping measurement %s: no address found", name)
                skipped["no_address"] += 1
                continue

            # Apply RAM/ROM base offset
            address = adjust_ram_address(address)
            
            # Get limits
            lower_limit = getattr(m, 'lowerLimit', 0.0) or 0.0
            upper_limit = getattr(m, 'upperLimit', 255.0) or 255.0
            
            # Get conversion method
            compu_method_name = getattr(m, 'conversionRef', None) or getattr(m, 'conversion', None)
            math, units = resolve_compu_method(session, compu_method_name)
            
            # Check for bit mask
            bitmask = getattr(m, 'bitMask', None)
            if bitmask is None:
                bitmask = getattr(m, 'bit_mask', None)
            # Extract actual value if it's a wrapper object
            if bitmask is not None:
                if hasattr(bitmask, 'mask'):
                    bitmask = int(bitmask.mask)
                elif hasattr(bitmask, 'value'):
                    bitmask = int(bitmask.value)
                else:
                    try:
                        bitmask = int(bitmask)
                    except (TypeError, ValueError):
                        bitmask = None
            
            # Check for array size
            array_size = getattr(m, 'arraySize', None)
            if array_size is None:
                array_size = getattr(m, 'array_size', None)
            # Extract actual value if it's a wrapper object
            if array_size is not None:
                if hasattr(array_size, 'number'):
                    array_size = int(array_size.number)
                elif hasattr(array_size, 'value'):
                    array_size = int(array_size.value)
                else:
                    try:
                        array_size = int(array_size)
                    except (TypeError, ValueError):
                        array_size = 1
            else:
                array_size = 1
            
            # Variant filter
            variant = getattr(m, "variantCoding", None) or getattr(m, "variant_coding", None)
            if args and args.variant and variant and args.variant not in str(variant):
                skipped["variant"] += 1
                continue

            # Determine category
            if bitmask is not None:
                category_idx = add_adx_category("Flags")
            elif array_size > 1:
                category_idx = add_adx_category("Arrays")
            else:
                category_idx = add_adx_category("Measurements")

            # Track statistics
            measurement_stats[datatype] += 1

            enum_labels = compu_enum_labels(session, compu_method_name)
            if not enum_labels and bitmask is not None:
                # Default boolean labels for flags
                enum_labels = {"0": "Off", "1": "On"}

            # Byte order per measurement
            meas_lsb_first = lsb_first
            if hasattr(m, "byteOrder") or hasattr(m, "byteorder"):
                meas_lsb_first = is_little_endian(m)

            # EVENT/DAQ based pollrate if available
            event_period = getattr(m, "maxRefresh", None)
            if event_period is not None:
                try:
                    pollrate = int(event_period)
                except Exception:
                    pollrate = default_pollrate

            # Per-measurement pollrate/event hint
            pollrate = getattr(m, "samplePeriod", None) or getattr(m, "sampleRate", None) or locals().get("pollrate", None)
            if pollrate is None:
                pollrate = default_pollrate

            # DAQ group / event ID if available
            daq_group = getattr(m, "event", None) or getattr(m, "eventId", None) or getattr(m, "event_id", None)

            # Variant / address mapping hint
            variant = getattr(m, "variantCoding", None) or getattr(m, "variant_coding", None)

            # If event maps to cycle/priority, override pollrate/priority
            priority = None
            if isinstance(daq_group, str) and daq_group in event_map:
                ev_info = event_map[daq_group]
                if ev_info.get("cycle") is not None:
                    try:
                        pollrate = int(ev_info["cycle"])
                    except Exception:
                        pass
                priority = ev_info.get("priority")
            
            # Bitmask math for flags (boolean/bitfields)
            bit_math = None
            if bitmask is not None and bitmask != 0:
                # find least significant set bit for shift
                shift = 0
                bm_tmp = bitmask
                while bm_tmp & 1 == 0:
                    bm_tmp >>= 1
                    shift += 1
                bit_math = f"((X & {hex(bitmask)}) >> {shift})"
                if not enum_labels:
                    enum_labels = compu_bit_enums(bitmask)

            # Build measurement data dict
            meas_data = {
                "name": name,
                "description": description,
                "address": hex(address),
                "datatype": datatype,
                "units": units,
                "min": lower_limit,
                "max": upper_limit,
                "math": bit_math or math,
                "bitmask": bitmask,
                "array_size": array_size,
                "pollrate": pollrate,
                "daq_group": daq_group,
                "variant": variant,
                "priority": priority,
            }
            
            # Create ADX channel
            create_adx_channel(root, meas_data, idx, category_idx, lsb_first=meas_lsb_first, enum_labels=enum_labels)
            processed_count += 1
            
        except Exception as e:
            logging.debug("Error processing measurement %s: %s", 
                         getattr(m, 'name', 'unknown'), e)
            skipped["error"] += 1
            continue
    
    logging.info("Processed %d measurements for ADX", processed_count)
    if skipped:
        logging.info("Skipped measurements: %s", dict(skipped))
    if args is not None and getattr(args, "strict", False) and sum(skipped.values()) > 0:
        raise RuntimeError("Strict mode: skipped measurements present")

    # Coverage summary
    total = processed_count + sum(skipped.values())
    logging.info("Measurement coverage: %d/%d (%.2f%%)", processed_count, total, (processed_count/total*100) if total else 0.0)
    
    # Print measurement statistics
    if measurement_stats:
        logging.info("")
        logging.info("=" * 60)
        logging.info("MEASUREMENT DATA TYPE STATISTICS:")
        logging.info("=" * 60)
        for dtype, count in sorted(measurement_stats.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  {dtype:30s}: {count:5d}")
        logging.info("=" * 60)
        logging.info(f"  {'TOTAL MEASUREMENTS':30s}: {sum(measurement_stats.values()):5d}")
        logging.info("=" * 60)
    
    return processed_count


def get_base_offset(session, desired="_ROM"):
    segs = session.query(model.MemorySegment).order_by(model.MemorySegment.address).all()
    if not segs:
        raise RuntimeError("No MemorySegments in A2L")

    for s in segs:
        if s.name == desired:
            return s.address

    logging.warning("Segment %r not found. Using %r", desired, segs[0].name)
    return segs[0].address


def get_segments(session):
    return session.query(model.MemorySegment).order_by(model.MemorySegment.address).all()


def process_all(session, root, header, args):
    # Get all characteristics from the DB
    chars = session.query(model.Characteristic).order_by(model.Characteristic.name).all()
    logging.info("Found %d characteristics", len(chars))

    # Global categories
    add_category(header, "Tables")
    add_category(header, "Axis")
    add_category(header, "Values")  # New category for scalar values

    processed_count = 0
    skipped_count = 0

    skipped_reasons: Counter = Counter()

    # Pre-compute characteristic -> categories from FUNCTION/GROUP if available
    char_categories: Dict[str, Set[str]] = {}
    function_descriptions: Dict[str, str] = {}  # function name -> description
    
    try:
        functions = session.query(model.Function).all()
        for f in functions:
            fn_name = getattr(f, "name", None)
            fn_desc = getattr(f, "longIdentifier", None)
            
            # Store function description
            if fn_name and fn_desc:
                function_descriptions[fn_name] = fn_desc
            
            # Map characteristics to functions
            defs = getattr(f, "def_characteristic", None) or getattr(f, "defCharacteristics", None) or []
            for ch in defs:
                # If ch is an object, get its name
                ch_name = getattr(ch, "name", str(ch))
                char_categories.setdefault(ch_name, set()).add(f"FUNC:{fn_name}")
            
            # Also check ref_characteristic if available
            refs = getattr(f, "ref_characteristic", None) or getattr(f, "refCharacteristics", None) or []
            for ch in refs:
                ch_name = getattr(ch, "name", str(ch))
                char_categories.setdefault(ch_name, set()).add(f"FUNC:{fn_name}")
    except Exception:
        pass
    try:
        groups = session.query(model.Group).all()
        for g in groups:
            gname = getattr(g, "name", None)
            members = getattr(g, "refCharacteristics", None) or getattr(g, "ref_characteristics", None) or []
            for ch in members:
                char_categories.setdefault(ch, set()).add(f"GRP:{gname}")
    except Exception:
        pass

    # Collect address ranges for overlap reporting
    z_ranges = []

    def skip(reason: str):
        skipped_reasons[reason] += 1

    name_filter: Optional[re.Pattern] = re.compile(args.name_filter) if args.name_filter else None

    for c in chars:
        try:
            ci = inspect.Characteristic(session, c.name)
        except Exception as e:
            logging.debug("Skipping %s due to error: %s", c.name, e)
            skipped_count += 1
            skip("inspect_error")
            continue

        if name_filter and not name_filter.search(c.name):
            skipped_count += 1
            skip("filter")
            continue

        # Get characteristic type
        ctype = get_characteristic_type(ci, c)
        char_type_stats[ctype] += 1

        # Record layout / endian / offset / index mode
        record_layout = getattr(ci.deposit, "recordLayout", None) or getattr(ci.deposit, "record_layout", None) or getattr(ci, "deposit", None)
        lsb_first = is_little_endian(record_layout) if record_layout is not None else True
        alignment_bits = get_alignment_bits(record_layout)
        base_addr_raw = align_address(ci.address + get_address_offset(record_layout), alignment_bits)

        # Extract FNC_VALUES index mode (COLUMN_DIR vs ROW_DIR)
        index_mode = get_fnc_values_index_mode(record_layout) or "ROW_DIR"

        # Z-axis (actual data) base info
        z_datatype = getattr(ci.deposit.fncValues, 'data_type', None)
        compu_name = getattr(ci.compuMethod, 'name', None)
        z_math, _ = resolve_compu_method(session, compu_name)
        z_enum = compu_enum_labels(session, compu_name)
        z_range = compu_range_labels(session, compu_name)
        if not z_range:
            z_range = compu_piecewise_labels(session, compu_name)
        if z_datatype in ("FLOAT32_IEEE", "FLOAT64_IEEE") or z_enum or z_range:
            z_math = "X"

        # Bitmask on characteristic -> decode flag
        z_bitmask = extract_bitmask(ci)
        if z_bitmask:
            shift = 0
            tmp = z_bitmask
            while tmp and (tmp & 1) == 0:
                tmp >>= 1
                shift += 1
            z_math = f"((X & {hex(z_bitmask)}) >> {shift})"
            if not z_enum:
                z_enum = compu_bit_enums(z_bitmask)

        alignment_bits = get_alignment_bits(record_layout)

        # Titles should be the variable name; put descriptive text in description
        title = c.name
        description = ci.longIdentifier or ci.displayIdentifier or ""
        
        # Add FORMAT string if available
        format_obj = getattr(c, "format", None) or getattr(ci, "format", None)
        if format_obj:
            # Extract formatString from the Format object
            format_str = getattr(format_obj, "formatString", None) or str(format_obj)
            if format_str and "formatString" not in format_str:  # Avoid showing the whole object
                description += f"\nFormat: {format_str}"

        # Variant filter
        variant = getattr(c, "variantCoding", None) or getattr(c, "variant_coding", None)
        if args.variant and variant and args.variant not in str(variant):
            skipped_count += 1
            skip("variant")
            continue

        # Determine category based on type
        if "VAL_BLK" in ctype or "VALUE" in ctype:
            category = "Values"
            if args.only_tables:
                skipped_count += 1
                skip("only_tables")
                continue
        else:
            category = "Tables"
            if args.only_values:
                skipped_count += 1
                skip("only_values")
                continue

        # Add function/group categories if present
        extra_cats = list(char_categories.get(c.name, []))
        for cat in extra_cats:
            # Add function description to category name if available
            if cat.startswith("FUNC:"):
                fn_name = cat.split(":", 1)[1]
                fn_desc = function_descriptions.get(fn_name)
                if fn_desc:
                    # Create enhanced category name with description
                    enhanced_cat = f"{cat}: {fn_desc}"
                    add_category(header, enhanced_cat)
                else:
                    add_category(header, cat)
            else:
                add_category(header, cat)

        td: Dict[str, Any] = {
            "title": title,
            "description": description,
            "category": category,
            "z": {
                "min": ci.lowerLimit,
                "max": ci.upperLimit,
                "address": hex(adjust_address(base_addr_raw)),
                "dataSize": z_datatype,
                "units": fix_degree(getattr(ci.compuMethod, 'unit', '')),
                "math": z_math,
                "enum": z_enum,
                "enum_range": z_range,
                "bitmask": z_bitmask,
                "lsb_first": lsb_first,
                "length": 1,  # Default length (columns)
                "rows": 1,    # Default rows
                "index_mode": index_mode,  # COLUMN_DIR or ROW_DIR
            },
        }

        # Handle VAL_BLK: set block length from NUMBER attribute
        if "VAL_BLK" in ctype:
            block_len = get_block_length(ci, c)
            td["z"]["length"] = block_len
            logging.debug(
                "Characteristic %s detected as VAL_BLK with NUMBER=%d",
                c.name, block_len
            )

        # Handle VALUE type (single scalar)
        elif "VALUE" in ctype:
            td["z"]["length"] = 1
            logging.debug("Characteristic %s detected as VALUE (scalar)", c.name)

        # Parse Record Layout for dynamic offset calculation
        layout_components = parse_record_layout(record_layout)
        component_offsets = {}
        current_rel_offset = 0
        
        # Get axis lengths (needed for size calculation)
        axes = ci.axisDescriptions
        x_len = 1
        y_len = 1
        if len(axes) > 0:
             x_len = int(getattr(axes[0], "maxAxisPoints", 1) or 1)
        if len(axes) > 1:
             y_len = int(getattr(axes[1], "maxAxisPoints", 1) or 1)

        for comp in layout_components:
            component_offsets[comp.name] = current_rel_offset
            
            size = 0
            if comp.relation == "NO_AXIS":
                size = get_data_size(comp.datatype)
            elif comp.relation == "AXIS_PTS":
                dsize = get_data_size(comp.datatype)
                if "_x" in comp.name:
                    size = x_len * dsize
                elif "_y" in comp.name:
                    size = y_len * dsize
            
            current_rel_offset += size

        # X axis
        if len(axes) > 0:
            x_info = axis_ref_to_dict(axes[0])

            if x_info is not None:
                # Handle inline axis (STD_AXIS DEPOSIT ABSOLUTE)
                if x_info.get("inline_axis"):
                    # Use calculated offset from Record Layout
                    # We look for axis_pts_x (standard) or fallback to heuristic if missing
                    # But for STD_AXIS, it should be in the layout.
                    
                    # Find the component name for X axis points
                    pts_comp = next((c for c in layout_components if c.relation == "AXIS_PTS" and "_x" in c.name), None)
                    
                    if pts_comp:
                        offset = component_offsets[pts_comp.name]
                        axis_addr = align_address(base_addr_raw + offset, alignment_bits)
                        x_info["address"] = hex(adjust_address(axis_addr))
                        
                        # Use datatype from Record Layout if available
                        x_info["dataSize"] = pts_comp.datatype
                        
                        logging.debug(
                            "Characteristic %s: X axis inline at 0x%X (offset %d) via %s",
                            c.name, axis_addr, offset, pts_comp.name
                        )
                    else:
                        logging.warning("Characteristic %s: Inline X axis but no AXIS_PTS_X in Record Layout", c.name)

                x_info["lsb_first"] = lsb_first

                td["x"] = x_info
                td["z"]["length"] = x_info["length"]
                td["description"] = (td["description"] or "") + f"\nX: {x_info['name']}"
            else:
                logging.debug(f"Skipping {c.name}: X axis not usable")
                skipped_count += 1
                skip("x_axis")
                continue

        # Y axis (if present)
        if len(axes) > 1:
            y_info = axis_ref_to_dict(axes[1])
            if y_info is not None:
                # Handle inline axis (STD_AXIS DEPOSIT ABSOLUTE)
                if y_info.get("inline_axis"):
                    pts_comp = next((c for c in layout_components if c.relation == "AXIS_PTS" and "_y" in c.name), None)
                    
                    if pts_comp:
                        offset = component_offsets[pts_comp.name]
                        axis_addr = align_address(base_addr_raw + offset, alignment_bits)
                        y_info["address"] = hex(adjust_address(axis_addr))
                        y_info["dataSize"] = pts_comp.datatype
                        
                        logging.debug(
                            "Characteristic %s: Y axis inline at 0x%X (offset %d) via %s",
                            c.name, axis_addr, offset, pts_comp.name
                        )
                    else:
                        logging.warning("Characteristic %s: Inline Y axis but no AXIS_PTS_Y in Record Layout", c.name)

                y_info["lsb_first"] = lsb_first

                td["y"] = y_info
                td["z"]["rows"] = y_info["length"]
                td["description"] += f"\nY: {y_info['name']}"
            else:
                logging.debug(f"Skipping {c.name}: Y axis not usable")
                skipped_count += 1
                skip("y_axis")
                continue

        # Handle COLUMN_DIR: Swap X and Y axes to match XDF row-major expectations
        if index_mode == "COLUMN_DIR" and len(axes) > 1:
            if "x" in td and "y" in td:
                # Swap axis assignments
                td["x"], td["y"] = td["y"], td["x"]
                # Swap length (cols) and rows in Z
                td["z"]["length"], td["z"]["rows"] = td["z"]["rows"], td["z"]["length"]
                # Update description to reflect swapped axes
                x_name = td["x"]["name"]
                y_name = td["y"]["name"]
                td["description"] = (td["description"] or "").replace(
                    f"\nX: {y_name}\nY: {x_name}",
                    f"\nX: {x_name}\nY: {y_name}"
                )
                logging.debug(
                    "Characteristic %s: COLUMN_DIR detected, swapped X/Y axes (X=%s cols=%d, Y=%s rows=%d)",
                    c.name, x_name, td["z"]["length"], y_name, td["z"]["rows"]
                )

        # Update Z axis address based on FNC_VALUES offset
        fnc_comp = next((c for c in layout_components if c.relation == "FNC_VALUES"), None)
        if fnc_comp:
            offset = component_offsets[fnc_comp.name]
            z_addr = align_address(base_addr_raw + offset, alignment_bits)
            td["z"]["address"] = hex(adjust_address(z_addr))
            logging.debug(
                "Characteristic %s: Z data at 0x%X (offset %d) via %s",
                c.name, z_addr, offset, fnc_comp.name
            )

        # Track address range for overlap check (if we know address and length)
        try:
            if td["z"].get("address"):
                addr_int = int(td["z"]["address"], 16)
                elem_bits = get_data_size(td["z"].get("dataSize")) * 8
                rows = int(td["z"].get("rows", 1))
                cols = int(td["z"].get("length", 1))
                stride_bits = None
                if td["z"].get("stride_bytes"):
                    stride_bits = td["z"]["stride_bytes"] * 8
                _, end_addr = addr_range(addr_int, elem_bits, cols * rows, alignment_bits, stride_bits)
                z_ranges.append((addr_int, end_addr))
                
                # Validation: check for address overlaps
                if args.validate:
                    size = get_data_size(td["z"].get("dataSize")) * cols * rows
                    validation_report.check_address_overlap(c.name, td["z"]["address"], size)
        except Exception:
            pass

        # Create main table
        t = SubElement(root, "XDFTABLE")
        t.set("uniqueid", new_unique_id())
        t.set("flags", "0x30")

        SubElement(t, "title").text = td["title"]
        SubElement(t, "description").text = td["description"]
        cats = [td["category"]]
        cats.extend(char_categories.get(c.name, []))
        add_table_categories(t, cats)

        # X axis handling
        if "x" in td:
            # Has real axis from axis descriptions
            real_axis(t, "x", td["x"])
            maybe_make_axis_table(root, td, "x")
        else:
            # No axis descriptions - create fake axis sized to Z length
            z_len = int(td["z"].get("length", 1))
            fake_axis(t, "x", z_len)

        # Y axis handling
        if "y" in td:
            real_axis(t, "y", td["y"])
            maybe_make_axis_table(root, td, "y")
        else:
            fake_axis(t, "y", 1)

        # Z axis (the actual data)
        z_addr = td["z"].get("address")
        if z_addr and z_addr in data_addresses_in_xdf:
            logging.debug("Duplicate data address detected for %s at %s", c.name, z_addr)
        data_addresses_in_xdf.add(z_addr)
        real_axis(t, "z", td["z"])

        # Enumeration mapping for discrete values
        if td["z"].get("enum") or td["z"].get("enum_range"):
            add_enumeration(t, td["z"].get("enum", {}), td["z"].get("enum_range", {}))
        processed_count += 1

    logging.info(f"Processed {processed_count} characteristics, skipped {skipped_count}")
    if skipped_reasons:
        logging.info("Skip reasons: %s", dict(skipped_reasons))
    total_chars = processed_count + skipped_count
    logging.info("Characteristic coverage: %d/%d (%.2f%%)", processed_count, total_chars, (processed_count/total_chars*100) if total_chars else 0.0)

    # Basic overlap/stride sanity: report duplicate data addresses and overlaps
    if data_addresses_in_xdf:
        dup_count = len(data_addresses_in_xdf) - len(set(data_addresses_in_xdf))
        if dup_count > 0:
            logging.warning("Detected %d duplicate Z data addresses", dup_count)

    if z_ranges:
        overlaps = region_overlaps(z_ranges)
        if overlaps > 0:
            logging.warning("Detected %d overlapping Z data ranges", overlaps)
    if args.strict and skipped_count > 0:
        raise RuntimeError("Strict mode: skipped items present")

    # Print characteristic type statistics
    if char_type_stats:
        logging.info("=" * 60)
        logging.info("CHARACTERISTIC TYPE STATISTICS:")
        logging.info("=" * 60)
        for ctype, count in sorted(char_type_stats.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  {ctype:30s}: {count:5d}")
        logging.info("=" * 60)
        logging.info(f"  {'TOTAL CHARACTERISTICS':30s}: {sum(char_type_stats.values()):5d}")
        logging.info("=" * 60)

    # Print axis type statistics
    if axis_stats:
        logging.info("")
        logging.info("=" * 60)
        logging.info("AXIS TYPE STATISTICS:")
        logging.info("=" * 60)
        for axis_type, count in sorted(axis_stats.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  {axis_type:30s}: {count:5d}")
        logging.info("=" * 60)
        logging.info(f"  {'TOTAL AXES':30s}: {sum(axis_stats.values()):5d}")
        logging.info("=" * 60)

        logging.info("")
        logging.info("Axis Type Explanations:")
        logging.info("  MEMORY_MAPPED             : Axes stored in memory (included in XDF)")
        logging.info("  FIX_AXIS_PAR              : Fixed axis with offset/shift")
        logging.info("  FIX_AXIS_PAR_DIST         : Fixed axis with distance")
        logging.info("  FIX_AXIS_PAR_LIST         : Fixed axis with explicit list")
        logging.info("  CURVE_AXIS_REF            : Reference to another curve's axis")
        logging.info("  DEPOSIT_ABSOLUTE_NO_PARAMS: Computed axes without layout (excluded)")
        logging.info("")
        logging.info("Note: Only MEMORY_MAPPED and FIX_* axes get full XDF support.")
        logging.info("      Purely computed axes without addresses are skipped.")
    
    return processed_count, len(functions) if 'functions' in locals() else 0


def parse_args():
    p = argparse.ArgumentParser(description="Convert A2L → XDF (+ optional ADX for measurements)")
    p.add_argument("a2l", help="Input A2L")
    p.add_argument("-o", "--output", help="Output XDF", default=None)
    p.add_argument("--adx", help="Output ADX file for measurements (optional)", default=None)
    p.add_argument("--no-adx", action="store_true", help="Skip ADX generation even if measurements exist")
    p.add_argument("--segment", help="Base ROM segment", default="_ROM")
    p.add_argument("--ram-segment", help="RAM segment for measurements", default=None)
    p.add_argument("--rom-base", type=lambda x: int(x, 0), help="Override ROM base address (hex or dec)")
    p.add_argument("--ram-base", type=lambda x: int(x, 0), help="Override RAM base address (hex or dec)")
    p.add_argument("--region-size", type=lambda x: int(x, 0), default=0x400000, help="XDF region size")
    p.add_argument("--title", help="Override XDF/ADX title", default=None)
    p.add_argument("--filter", dest="name_filter", help="Regex to include only matching CHARACTERISTIC names", default=None)
    p.add_argument("--only-values", action="store_true", help="Export only VALUE/VAL_BLK types")
    p.add_argument("--only-tables", action="store_true", help="Export only CURVE/MAP types")
    p.add_argument("--strict", action="store_true", help="Fail on skipped items due to missing data")
    p.add_argument("--protocol", choices=["CCP", "KWP", "UDS"], default="CCP", help="ADX protocol")
    p.add_argument("--pollrate", type=int, default=100, help="Default ADX poll rate (ms)")
    p.add_argument("--big-endian", action="store_true", help="Set ADX channels to MSB-first")
    p.add_argument("--force-db", action="store_true", help="Rebuild .a2ldb cache if it already exists")
    p.add_argument("--variant", help="Variant name to include (if variantCoding present)", default=None)
    p.add_argument("--validate", action="store_true", help="Enable validation and diagnostics")
    p.add_argument("--log-level", default=None, help="Logging level (DEBUG,INFO,WARNING,ERROR)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    level = args.log_level.upper() if args.log_level else ("DEBUG" if args.verbose else "INFO")
    logging.basicConfig(
        level=level,
        format='%(levelname)s:%(name)s:%(message)s'
    )

    # Reset global state for clean runs
    global BASE_OFFSET, RAM_BASE_OFFSET, categories, axis_addresses_in_xdf
    global axis_stats, char_type_stats, measurement_stats
    BASE_OFFSET = 0
    RAM_BASE_OFFSET = 0
    categories = []
    axis_addresses_in_xdf = set()
    axis_stats = Counter()
    char_type_stats = Counter()
    measurement_stats = Counter()

    db = DB()
    a2l_base = args.a2l
    try:
        session = db.import_a2l(
            a2l_base,
            remove_existing=args.force_db,
            loglevel=level,
        )
    except OSError as e:
        if "already exists" in str(e) and not args.force_db:
            logging.info("A2L DB cache exists; retrying with --force-db behavior")
            session = db.import_a2l(
                a2l_base,
                remove_existing=True,
                loglevel=level,
            )
        else:
            raise

    # Log library version for reproducibility
    try:
        logging.info("pya2l version: %s", importlib.metadata.version("pya2ldb"))
    except importlib.metadata.PackageNotFoundError:
        try:
            logging.info("pya2l version: %s", importlib.metadata.version("pya2l"))
        except importlib.metadata.PackageNotFoundError:
            logging.info("pya2l version: unknown (metadata missing)")

    BASE_OFFSET = args.rom_base if args.rom_base is not None else get_base_offset(session, args.segment)
    logging.info("BASE_OFFSET (ROM) = 0x%X", BASE_OFFSET)

    # RAM base: explicit override, ram-segment, or fallback to ROM
    if args.ram_base is not None:
        RAM_BASE_OFFSET = args.ram_base
    elif args.ram_segment:
        try:
            RAM_BASE_OFFSET = get_base_offset(session, args.ram_segment)
        except RuntimeError:
            RAM_BASE_OFFSET = BASE_OFFSET
    else:
        RAM_BASE_OFFSET = BASE_OFFSET
    logging.info("RAM_BASE_OFFSET = 0x%X", RAM_BASE_OFFSET)

    # Load ADDRESS_MAPPING from A2L text
    global address_mappings
    address_mappings = parse_address_mappings(a2l_base)

    # Generate XDF for CHARACTERISTICs
    xdf_title = args.title or a2l_base
    root, header = xdf_root_with_configuration(xdf_title, region_size=args.region_size)

    # Add regions for all memory segments if present
    try:
        SEGMENTS = get_segments(session)
        if SEGMENTS:
            # remove the default region added above
            for child in list(header.findall("REGION")):
                header.remove(child)
            for s in SEGMENTS:
                region = SubElement(header, "REGION")
                region.set("type", "0xFFFFFFFF")
                region.set("startaddress", hex(int(s.address)))
                region.set("size", hex(int(s.size if hasattr(s, 'size') and s.size else args.region_size)))
                region.set("regionflags", "0x0")
                region.set("name", getattr(s, 'name', 'SEG'))
                region.set("desc", getattr(s, 'description', getattr(s, 'name', 'Segment')))
    except Exception:
        SEGMENTS = []

    processed_count, function_count = process_all(session, root, header, args)

    out = args.output or f"{a2l_base}.xdf"
    write_pretty_xml(root, out)
    logging.info("Wrote XDF: %s", out)

    # Generate ADX for MEASUREMENTs (unless disabled)
    if not args.no_adx:
        adx_out = args.adx or f"{a2l_base}.adx"
        adx_title = args.title or a2l_base
        adx_root, adx_header = adx_root_with_configuration(adx_title, protocol=args.protocol, pollrate=args.pollrate)
        meas_count = process_measurements(session, adx_root, adx_header, lsb_first=not args.big_endian, default_pollrate=args.pollrate, args=args)
        
        if meas_count > 0:
            write_pretty_xml(adx_root, adx_out)
            logging.info("Wrote ADX: %s (%d channels)", adx_out, meas_count)
        else:
            logging.info("No measurements found - ADX file not created")
    
    # Print validation summary if enabled
    if args.validate:
        # Add summary info messages
        validation_report.add_info(f"Processed {len(validation_report.address_map)} unique addresses")
        validation_report.add_info(f"Total characteristics: {sum(char_type_stats.values())}")
        if function_count > 0:
            validation_report.add_info(f"Functions with descriptions: {function_count}")
        
        # Print the summary
        validation_report.print_summary()


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------- #
# Programmatic API for GUI/other callers                                      #
# --------------------------------------------------------------------------- #
from dataclasses import dataclass


@dataclass
class ConvertOptions:
    a2l: str
    output: Optional[str] = None
    adx: Optional[str] = None
    no_adx: bool = False
    segment: str = "_ROM"
    ram_segment: Optional[str] = None
    rom_base: Optional[int] = None
    ram_base: Optional[int] = None
    region_size: int = 0x400000
    title: Optional[str] = None
    name_filter: Optional[str] = None
    only_values: bool = False
    only_tables: bool = False
    strict: bool = False
    protocol: str = "CCP"
    pollrate: int = 100
    big_endian: bool = False
    force_db: bool = False
    variant: Optional[str] = None
    log_level: str = "INFO"
    verbose: bool = False


def convert(opts: ConvertOptions):
    """
    Programmatic entry point mirroring CLI main.
    Returns dict with paths and counts.
    """
    import argparse as _argparse
    args = _argparse.Namespace(**opts.__dict__)
    # Run through main logic by reusing main sections
    db = DB()
    level = opts.log_level.upper() if opts.log_level else ("DEBUG" if opts.verbose else "INFO")
    logging.basicConfig(level=level, format='%(levelname)s:%(name)s:%(message)s')

    # Reset globals
    global BASE_OFFSET, RAM_BASE_OFFSET, categories, axis_addresses_in_xdf, axis_stats, char_type_stats, measurement_stats, data_addresses_in_xdf, address_mappings
    BASE_OFFSET = 0
    RAM_BASE_OFFSET = 0
    categories = []
    axis_addresses_in_xdf = set()
    axis_stats = Counter()
    char_type_stats = Counter()
    measurement_stats = Counter()
    data_addresses_in_xdf = set()

    a2l_base = opts.a2l
    address_mappings = parse_address_mappings(a2l_base)

    try:
        session = db.import_a2l(a2l_base, remove_existing=opts.force_db, loglevel=level)
    except OSError as e:
        if "already exists" in str(e):
            session = db.import_a2l(a2l_base, remove_existing=True, loglevel=level)
        else:
            raise

    BASE_OFFSET = opts.rom_base if opts.rom_base is not None else get_base_offset(session, opts.segment)
    if opts.ram_base is not None:
        RAM_BASE_OFFSET = opts.ram_base
    elif opts.ram_segment:
        try:
            RAM_BASE_OFFSET = get_base_offset(session, opts.ram_segment)
        except RuntimeError:
            RAM_BASE_OFFSET = BASE_OFFSET
    else:
        RAM_BASE_OFFSET = BASE_OFFSET

    xdf_title = opts.title or a2l_base
    root, header = xdf_root_with_configuration(xdf_title, region_size=opts.region_size)
    try:
        SEGMENTS = get_segments(session)
        if SEGMENTS:
            for child in list(header.findall("REGION")):
                header.remove(child)
            for s in SEGMENTS:
                region = SubElement(header, "REGION")
                region.set("type", "0xFFFFFFFF")
                region.set("startaddress", hex(int(s.address)))
                region.set("size", hex(int(s.size if hasattr(s, 'size') and s.size else opts.region_size)))
                region.set("regionflags", "0x0")
                region.set("name", getattr(s, 'name', 'SEG'))
                region.set("desc", getattr(s, 'description', getattr(s, 'name', 'Segment')))
    except Exception:
        SEGMENTS = []

    process_all(session, root, header, args)
    out = opts.output or f"{a2l_base}.xdf"
    write_pretty_xml(root, out)

    adx_out = None
    meas_count = 0
    if not opts.no_adx:
        adx_out = opts.adx or f"{a2l_base}.adx"
        adx_title = opts.title or a2l_base
        adx_root, adx_header = adx_root_with_configuration(adx_title, protocol=opts.protocol, pollrate=opts.pollrate)
        meas_count = process_measurements(session, adx_root, adx_header, lsb_first=not opts.big_endian, default_pollrate=opts.pollrate, args=args)
        if meas_count > 0:
            write_pretty_xml(adx_root, adx_out)
        else:
            adx_out = None

    return {"xdf": out, "adx": adx_out, "measurements": meas_count}
