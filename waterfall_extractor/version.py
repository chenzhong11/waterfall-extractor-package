"""Version metadata for the waterfall extractor."""

PACKAGE_VERSION = "1.0.0"
ALGORITHM_ID = "spectrum_waterfall_v1"
ALGORITHM_VERSION = "A-1.0.0"
FEATURE_SCHEMA_VERSION = "2026.07"

CREATOR = "位豪"
LAST_EDIT_DATE = "2026-06-29"


def get_version_info() -> dict[str, str]:
    """返回版本信息字典，便于集成方记录算法来源。"""
    return {
        "package_version": PACKAGE_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "algorithm_version": ALGORITHM_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "creator": CREATOR,
        "last_edit_date": LAST_EDIT_DATE,
    }
