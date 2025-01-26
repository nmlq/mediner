import pytest
import pandas
import json
import os


LABEL_STUDIO_EXPORT_JSON = json.loads('[\n  {\n    "id": 1,\n    "annotations": [\n      {\n        "id": 1,\n        "completed_by": 1,\n        "result": [\n          {\n            "value": {\n              "start": 647,\n              "end": 648,\n              "text": "5",\n              "labels": [\n                "PIRADS Score"\n              ]\n            },\n            "id": "vDe6XogBUJ",\n            "from_name": "label",\n            "to_name": "text",\n            "type": "labels",\n            "origin": "manual"\n          },\n          {\n            "value": {\n              "start": 293,\n              "end": 308,\n              "text": "4.3 x 2.9 x 3.0",\n              "labels": [\n                "Prostate dimensions"\n              ]\n            },\n            "id": "okRZFmKL-O",\n            "from_name": "label",\n            "to_name": "text",\n            "type": "labels",\n            "origin": "manual"\n          }\n        ],\n        "was_cancelled": false,\n        "ground_truth": false,\n        "created_at": "2025-01-16T22:35:45.345934Z",\n        "updated_at": "2025-01-16T22:35:45.345964Z",\n        "draft_created_at": "2025-01-16T22:34:53.465205Z",\n        "lead_time": 114.708,\n        "prediction": {},\n        "result_count": 2,\n        "unique_id": "302efa25-e354-45d8-aa57-e3d7fb288916",\n        "import_id": null,\n        "last_action": null,\n        "task": 1,\n        "project": 1,\n        "updated_by": 1,\n        "parent_prediction": null,\n        "parent_annotation": null,\n        "last_created_by": null\n      }\n    ],\n    "file_upload": "8314a79b-Copy_of_De-ID_MRI_Sample_1_13_2025-label-studio.json",\n    "drafts": [],\n    "predictions": [],\n    "data": {\n      "text": "\\rMRI PELVIS W&W/O CONTRAST\\r\\rClinical Indication: Elevated PSA and microhematuria.\\r\\rTechnique: Multi-parametric prostate MRI using 9 cc of Gadavist\\rintravenous contrast.  \\r\\rComparison: None.\\r\\rBiopsy: None.\\r\\rPSA:  7.5 ng/mL on 4/11/2022\\r\\rPSA density:  0.42 ng/mL/cc\\r\\rThe prostate gland measures 4.3 x 2.9 x 3.0 cm (18 cc). \\r\\rNo focal suspicious T2 signal within the transitional zone. \\r\\rScattered mildly heterogeneous T2 signal within the peripheral\\rzone bilaterally without focal restricted diffusion, likely\\rprostatitis/scarring.\\r\\rIn the left peripheral zone from the apex almost to the base from\\r2-5 o\'clock there is a 1.1 x 0.5 x 1.9 cm PI-RADS 5 lesion. This\\rbroad-based interface with the capsule and adjacent tissue is\\rsuspicious for extraprostatic extension and subtle focal bulge\\rtoward the left neurovascular bundle in the apex is suggested\\r(series 6-image 14).\\r\\rThere are is T2 hypointensity (series 6 - images 10/15), ADC map\\rhypointensity and high b-value DWI (series 8/102 - images 10/14)\\rand there is focal and early enhancement.  \\r\\rSeminal vesicles are symmetric. \\r\\rRecto- prostatic fat and fascial planes are intact.\\r\\rNo focal bladder wall thickening. The urinary bladder is markedly\\rdistended even after several attempts at urination.\\r\\rNo enlarged pelvic lymph nodes.\\r\\rImaged bowel and rectum without wall thickening. No free fluid. \\r\\rNo suspicious bone marrow signal or enhancement.\\r\\r"\n    },\n    "meta": {},\n    "created_at": "2025-01-16T22:29:00.737559Z",\n    "updated_at": "2025-01-16T22:35:45.423742Z",\n    "inner_id": 1,\n    "total_annotations": 1,\n    "cancelled_annotations": 0,\n    "total_predictions": 0,\n    "comment_count": 0,\n    "unresolved_comment_count": 0,\n    "last_comment_updated_at": null,\n    "project": 1,\n    "updated_by": 1,\n    "comment_authors": []\n  }\n]')  # noqa: E501


@pytest.fixture(scope="session")
def mock_input_data() -> list[dict]:
    return [
        {
            "Sta3n": 123,
            "SiteAccessionNumber": "NULL",
            "ReportText": "Some text"
        },
        {
            "Sta3n": 456,
            "SiteAccessionNumber": "111-222222-3333",
            "ReportText": "Some more text"
        }
    ]


@pytest.fixture(scope="session")
def mock_input_df(mock_input_data) -> pandas.DataFrame:
    return pandas.DataFrame(mock_input_data)


@pytest.fixture(scope="session")
def mock_input_csv(mock_input_df, tmp_path_factory):
    mock_input_csv_path = tmp_path_factory.mktemp(
        "mock-data-mediner"
    ) / "mock-input.csv"
    mock_input_df.to_csv(mock_input_csv_path, index=False)
    return str(mock_input_csv_path)


@pytest.fixture(scope="session")
def mock_label_studio_export_json(mock_input_df, tmp_path_factory):
    mock_label_studio_export_json_path = tmp_path_factory.mktemp(
        "mock-data-mediner"
    ) / "mock-label-studio-export.json"
    with open(mock_label_studio_export_json_path, 'w') as jf:
        jf.write(json.dumps(LABEL_STUDIO_EXPORT_JSON))
    return str(mock_label_studio_export_json_path)


@pytest.fixture(scope="session")
def test_config_filename():
    dirname = os.path.dirname(os.path.abspath(__file__))
    test_config_filename = f"{dirname}/test_config.cfg"
    return test_config_filename
