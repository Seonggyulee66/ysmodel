from opencood.data_utils.datasets.camera_only.base_camera_dataset import BaseCameraDataset
from opencood.data_utils.datasets.camera_only.late_fusion_dataset import CamLateFusionDataset
from opencood.data_utils.datasets.camera_only.intermediate_fusion_dataset import CamIntermediateFusionDataset
from opencood.data_utils.datasets.camera_only.intermediate_fusion_dataset_per_scenario import CamIntermediateFusionDataset_per_scenario
from opencood.data_utils.datasets.intermediate_fusion_dataset_v2 import IntermediateFusionDatasetV2
__all__ = {
    'BaseCameraDataset': BaseCameraDataset,
    'CamLateFusionDataset': CamLateFusionDataset,
    'CamIntermediateFusionDataset': CamIntermediateFusionDataset,
    'IntermediateFusionDatasetV2' : IntermediateFusionDatasetV2,
    'CamIntermediateFusionDataset_per_scenario' : CamIntermediateFusionDataset_per_scenario,
}

# the final range for evaluation
GT_RANGE = [-140, -40, -3, 140, 40, 1]
CAMERA_GT_RANGE = [-50, -50, -3, 50, 50, 1]
# The communication range for cavs
COM_RANGE = 100

def build_dataset(dataset_cfg, visualize=False, train=True, validate=False):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in ['LateFusionDataset',
                            'EarlyFusionDataset',
                            'IntermediateFusionDataset',
                            'CamLateFusionDataset',
                            'CamIntermediateFusionDataset',
                            'CamIntermediateFusionDataset_per_scenario',
                            'BaseCameraDataset',
                            'IntermediateFusionDatasetV2'], error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        validate=validate
    )

    return dataset
