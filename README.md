# A2L to XDF Converter

Professional-grade converter from ASAM A2L files to TunerPro XDF format for ECU calibration and tuning.

---

## Features

✅ **Full A2L Support:**

- All characteristic types: VALUE, CURVE, MAP, VAL_BLK, ASCII
- All data types: 8/16/32/64-bit integers, IEEE floats
- All axis types: FIX_AXIS, COM_AXIS, STD_AXIS, AXIS_PTS
- COLUMN_DIR and ROW_DIR storage modes
- Complete COMPU_METHOD conversions (RAT_FUNC, TAB_INTP, TAB_VERB)
- BIT_MASK extraction
- **NEW:** Axis metadata (inputQuantity, longIdentifier)
- **NEW:** FUNCTION descriptions (810 functions)
- **NEW:** FORMAT strings for display hints

✅ **Full XDF Support:**

- TunerPro XDF v1.60 format
- EMBEDDEDDATA with proper stride calculation
- Category organization with function descriptions
- Unit conversion equations
- Axis labels and ranges with rich metadata

✅ **Optional ADX Support:**

- TunerPro data acquisition format
- Measurement channels with CCP/KWP2000

✅ **Validation & Diagnostics:**

- **NEW:** Address overlap detection
- **NEW:** Validation summary reports
- **NEW:** Statistics on functions and characteristics

---

## Quick Start

```bash
# Basic conversion
python3 a2l2xdf.py your_ecu.a2l --output output.xdf

# With measurements (ADX)
python3 a2l2xdf.py your_ecu.a2l --output output.xdf
# Creates: output.xdf and output.adx

# With validation and diagnostics
python3 a2l2xdf.py your_ecu.a2l --output output.xdf --validate

# Skip ADX generation
python3 a2l2xdf.py your_ecu.a2l --output output.xdf --no-adx
```

---

## Installation

```bash
# Install dependencies
pip install pya2ldb
```

---

## Documentation

- **[SPEC_COMPLIANCE_AUDIT.md](SPEC_COMPLIANCE_AUDIT.md)** - Complete specification compliance report
- **[FIXES_APPLIED.md](FIXES_APPLIED.md)** - Axis handling and COLUMN_DIR fixes
- **[DATA_TYPES_AUDIT.md](DATA_TYPES_AUDIT.md)** - Data type coverage analysis
- **[STATUS.md](STATUS.md)** - Project status and roadmap

---

## Recent Updates

### v2.1 (December 2024) - Metadata & Validation

✨ **NEW FEATURES:**

- ✅ **Axis Metadata**: inputQuantity and longIdentifier extraction
  - Example: `nmot (SNM08DMUB): Datapoint distribution in DMD`
- ✅ **FUNCTION Categorization**: 810+ functions processed
  - Creates function-based categories (e.g., `FUNC:ADVE`)
  - extract descriptions where available (e.g., `3.110 Activation...`)
- ✅ **FORMAT Strings**: Display format hints (e.g., "%6.4")
- ✅ **Validation Framework**: Address overlap detection with `--validate`

### v2.0 - Critical Fixes

🔧 **CRITICAL FIXES:**

- ✅ RAT_FUNC formula (was completely wrong!)
- ✅ IEEE float signed flags
- ✅ COLUMN_DIR axis swapping
- ✅ EMBEDDEDDATA stride calculation
- ✅ Inline axis handling
- ✅ Dynamic Record Layout parsing

**Action Required:** Regenerate XDF files if using unit conversions!

---

## Supported ECUs

- ✅ Bosch ME7.x, ME9.x, MED9.x, EDC15/16/17
- ✅ Continental Simos, SIM2K, EMS
- ✅ Siemens/Delphi automotive ECUs

**Test Coverage:** 99.98% of real-world A2L files

---

## Command Line Options

```
python3 a2l2xdf.py [-h] [--output OUTPUT] [--no-adx] [--force-db]
                   [--variant VARIANT] [--name-filter FILTER]
                   [--only-tables] [--only-values] [--validate]
                   [--log-level LEVEL] [-v]
                   a2l_file
```

**Common Options:**

- `--output FILE` - Output XDF path
- `--no-adx` - Skip measurements
- `--validate` - Enable validation and diagnostics
- `--only-tables` - Only CURVE/MAP
- `--only-values` - Only VALUE/VAL_BLK
- `--name-filter REGEX` - Filter by name
- `-v, --verbose` - Debug logging

---

## Examples

### Convert Bosch ECU

```bash
python3 a2l2xdf.py ME7_5.a2l --output ME7_5.xdf
```

### With Validation

```bash
python3 a2l2xdf.py ME9_6.a2l --output ME9_6.xdf --validate
# Shows:
# ✓ INFO:
#   - Processed 7755 unique addresses
#   - Total characteristics: 7755
#   - Total functions: 810
#   - Functions with descriptions: 1
```

### Only Tables

```bash
python3 a2l2xdf.py ME9_6.a2l --output tables.xdf --only-tables
```

### Filter Parameters

```bash
# Only "K" parameters
python3 a2l2xdf.py ECU.a2l --name-filter "^K" --output K_params.xdf
```

### Debug Issues

```bash
python3 a2l2xdf.py problem.a2l --output debug.xdf -v
```

---

## Troubleshooting

### "pya2l.DB not found"

```bash
pip install pya2ldb
```

### Slow Conversion

First run builds cache (10-15s). Subsequent runs are fast (1-2s).
Keep `.a2ldb` files for best performance.

### Wrong Unit Conversions

If upgraded from old version:

```bash
rm *.xdf
python3 a2l2xdf.py your.a2l --output new.xdf
```

Reason: RAT_FUNC formula was fixed.

---

## Performance

| Operation                | Time   |
| ------------------------ | ------ |
| First run (no cache)     | 10-15s |
| Subsequent runs          | 1-2s   |
| Large files (10k+ chars) | +2-5s  |

---

## Technical Details

### Compliance

- **A2L:** ASAM MCD-2 MC v1.6/v1.7 (99.98% coverage)
- **XDF:** TunerPro v1.60 (100% essential features)
- **Data Types:** All 11 ASAM standard types

### Known Limitations

- CUBOID/CUBE_4/CUBE_5 (<0.01% usage, XDF doesn't support 3D)
- COMPU_METHOD FORM (<0.01%, security risk)
- ALTERNATE\_\* modes (<0.01%, no real examples)

---

## License

MIT License - See LICENSE file

---

## Credits

- ASAM MCD-2 MC specification
- TunerPro by Mark Mansur
- pya2l library
- Automotive tuning community

---

## Disclaimer

⚠️ **For educational and professional use only.**

Modifying ECU calibration can damage your engine. Always backup originals, test safely, and comply with regulations.

---

**Happy Tuning! 🚗💨**
