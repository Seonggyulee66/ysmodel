"""
Fusion for intermediate level (camera)
"""
from collections import OrderedDict

import numpy as np
import torch

import opencood
from opencood.data_utils.datasets.camera_only import base_camera_dataset
from opencood.utils import common_utils

class CamIntermediateFusionDataset_per_scenario(base_camera_dataset.BaseCameraDataset):
    def __init__(self, params, visualize, train=True, validate=False):
        super(CamIntermediateFusionDataset_per_scenario, self).__init__(params,
                                                           visualize,
                                                           train,
                                                           validate)
        self.visible = params['train_params']['visible']
        self.scenario_list = self.build_scenario_index()
    
    ## 이전과는 다르게 scenario별로 처리하기위해 이전의 tick 기준 __len__을 override
    def __len__(self):
        return len(self.scenario_list)
        
    def build_scenario_index(self):
        scenario_list = []
        for scenario_id in self.scenario_idx_list:
            timestamps_list = self.get_tick_indices_for_scenario(scenario_id)    ## ['000069', '000071', '000073', '000075',,,,,]
            scenario_list.append({
                'scenario_id' : scenario_id,
                'timestamps_list' : timestamps_list,
                'tick_indices' : range(len(timestamps_list))
            })
        return scenario_list
        
    def __getitem__(self, idx):
        
        scenario = self.scenario_list[idx]
        scenario_id = scenario['scenario_id']           ## [0,1,2,3,4,,,,]
        timestamps_list = scenario['timestamps_list']   ## ['000069', '000071', '000073', '000075',,,,,]
        tick_indices = scenario['tick_indices']         ## range [0,1,2,3,4,5,,,,,]
        
        scenario_data = []
        for tick_idx in tick_indices:
            data_sample = self.get_sample_random(
                (scenario_id, tick_idx)
                )       ## data_sample: odict_keys(['659', '641', '650'])

            processed_tick = self.process_single_tick(data_sample)  ## data_sample에 있는 

            scenario_data.append(processed_tick)
            
        print(f"Scenario id ({scenario['scenario_id']}) has been finished")
            
        return scenario_data            ## [tick0_dict, tick1_dict, tick2_dict,....]
    
    def process_single_tick(self,data_sample):
        cav_list = list(data_sample.keys())
        
        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = OrderedDict()

        ego_id = -999
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in data_sample.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(data_sample.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -999
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = \
            self.get_pairwise_transformation(data_sample,
                                             self.params['train_params']['max_cav'])

        # Final shape: (L, M, H, W, 3)
        camera_data = []
        # (L, M, 3, 3)
        camera_intrinsic = []
        # (L, M, 4, 4)
        camera2ego = []

        # (max_cav, 4, 4)
        transformation_matrix = []
        # (1, H, W)
        gt_static = []
        # (1, h, w)
        gt_dynamic = []
        
        scenario_id = []
        agent_true_loc=[]
        # distance_list = []
        single_bev_image = []
        timestamp_key= []
        
        # loop over all CAVs to process information
        for cav_id, selected_cav_base in data_sample.items():
            ##  selected_cav_base : base dataset format
            ##  odict_keys(['ego', 'time_delay', 'camera_params', 'params', 'camera_np', 'bev_dynamic.png', 'bev_static.png', 'bev_lane.png', 'bev_visibility.png', 'bev_visibility_corp.png', 'object_bbx_cav', 'object_id', 'object_bbx_ego', 'object_bbx_ego_mask'])
            distance = common_utils.cav_distance_cal(selected_cav_base,
                                                     ego_lidar_pose)
            # distance_list.append(distance)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = \
                self.get_single_cav(selected_cav_base)
            
            camera_data.append(selected_cav_processed['camera']['data'])    ## camera_data: (M, H, W, 3)
            camera_intrinsic.append(
                selected_cav_processed['camera']['intrinsic'])
            camera2ego.append(
                selected_cav_processed['camera']['extrinsic'])
            transformation_matrix.append(
                selected_cav_processed['transformation_matrix'])
            scenario_id.append(selected_cav_processed['scenario_id'])
            agent_true_loc.append(selected_cav_processed['agent_true_loc'])
            single_bev_image.append(selected_cav_processed['single_dynamic_bev'])
            timestamp_key.append(selected_cav_processed['timestamp_key'])
            
            if cav_id == ego_id:
                gt_dynamic.append(
                    selected_cav_processed['gt']['dynamic_bev'])        ## selected_cav_processed['gt']['dynamic_bev']: (256,256)
                gt_static.append(
                    selected_cav_processed['gt']['static_bev'])

        # stack all agents together
        camera_data = np.stack(camera_data)         ## camera_data: (L, M, H, W, 3)
        camera_intrinsic = np.stack(camera_intrinsic)
        camera2ego = np.stack(camera2ego)
        agent_true_loc = np.stack(agent_true_loc)
        # distance_stack = np.stack(distance_list)
        single_bev_image = np.stack(single_bev_image)
        timestamp_key = np.stack(timestamp_key)

        gt_dynamic = np.stack(gt_dynamic)
        gt_static = np.stack(gt_static)

        # padding
        transformation_matrix = np.stack(transformation_matrix)
        padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(
                                               transformation_matrix), 1, 1))
        transformation_matrix = np.concatenate(
            [transformation_matrix, padding_eye], axis=0)

        processed_data_dict['ego'].update({
            'transformation_matrix': transformation_matrix,
            'pairwise_t_matrix': pairwise_t_matrix,
            'camera_data': camera_data,
            'camera_intrinsic': camera_intrinsic,
            'camera_extrinsic': camera2ego,
            'gt_dynamic': gt_dynamic,
            'gt_static': gt_static,
            'scenario_id': scenario_id,
            'agent_true_loc' : agent_true_loc,
            'cav_list' : cav_list,
            # 'dist_to_ego' : distance_stack,
            'single_bev_imgae' : single_bev_image,
            'timestamp_key' : timestamp_key})

        return processed_data_dict

    @staticmethod
    def get_pairwise_transformation(base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        # default are identity matrix
        pairwise_t_matrix[:, :] = np.identity(4)

        # return pairwise_t_matrix

        t_list = []

        # save all transformation matrix in a list in order first.
        for cav_id, cav_content in base_data_dict.items():
            t_list.append(cav_content['params']['transformation_matrix'])

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i == j:
                    continue
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix

    def get_single_cav(self, selected_cav_base):
        """
        Process the cav data in a structured manner for intermediate fusion.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = OrderedDict()
        selected_cav_processed.update({
            'scenario_id': selected_cav_base['scenario_id']
        })
        selected_cav_processed.update({
            'agent_true_loc': selected_cav_base['agent_true_loc']
        })
        selected_cav_processed.update({
            'timestamp_key': selected_cav_base['timestamp_key']
        })
        # update the transformation matrix
        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']
        selected_cav_processed.update({
            'transformation_matrix': transformation_matrix
        })

        selected_cav_processed.update({'single_dynamic_bev' : self.post_processor.generate_label(selected_cav_base['bev_dynamic.png'])})

        # for intermediate fusion, we only need ego's gt
        if selected_cav_base['ego']:
            # process the groundtruth
            if self.visible:
                dynamic_bev = \
                    self.post_processor.generate_label(
                        selected_cav_base['bev_visibility_corp.png'])
            else: 
                dynamic_bev = \
                    self.post_processor.generate_label(
                        selected_cav_base['merged_bev_dynamic.png'])
            road_bev = \
                self.post_processor.generate_label(
                    selected_cav_base['merged_bev_static.png'])
            lane_bev = \
                self.post_processor.generate_label(
                    selected_cav_base['merged_bev_lane.png'])
            
            static_bev = self.post_processor.merge_label(road_bev, lane_bev)

            gt_dict = {'static_bev': static_bev,
                       'dynamic_bev': dynamic_bev,}                  ## merged BEV

            selected_cav_processed.update({'gt': gt_dict})
 
        all_camera_data = []
        all_camera_origin = []
        all_camera_intrinsic = []
        all_camera_extrinsic = []

        # preprocess the input rgb image and extrinsic params first
        for camera_id, camera_data in selected_cav_base['camera_np'].items():
            all_camera_origin.append(camera_data)
            camera_data = self.pre_processor.preprocess(camera_data)
            camera_intrinsic = \
                selected_cav_base['camera_params'][camera_id][
                    'camera_intrinsic']
            cam2ego = \
                selected_cav_base['camera_params'][camera_id][
                    'camera_extrinsic_to_ego']

            all_camera_data.append(camera_data)
            all_camera_intrinsic.append(camera_intrinsic)
            all_camera_extrinsic.append(cam2ego)

        camera_dict = {
            'origin_data': np.stack(all_camera_origin),
            'data': np.stack(all_camera_data),
            'intrinsic': np.stack(all_camera_intrinsic),
            'extrinsic': np.stack(all_camera_extrinsic)
        }

        selected_cav_processed.update({'camera': camera_dict})

        return selected_cav_processed


    ## 이전 : Dataset에서 tick 단위별 data 처리 || Sampler에서 tick 단위 분배 || Batchsampler에서 batch_size로 묶음 || colllate_fn에서 batch size에 맞게 합쳐서 tensor 생성
    ## 현재 : Dataset에서 scenario 전체 tick 반환 || Sampler에서 scenario index 단위 분배 || BatchSampler 없음 || collate_fn에서 시퀀스 전체를 합침 
    ## 현재 Collate_fn의 역할 : tick들을 순회하면서 동일 field끼리 Concate -> Stack -> Tensor로 만듬  || sequence를 [num_ticks, ...] shape으로 유지해주는 역할을 수행
    def collate_batch(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        
        assert len(batch) == 1      ## scenario base는 batch 사이즈가 1이어야함
        
        scenario_sequence = batch[0]    ## batch = [ [tick0_dict, tick1_dict, tick2_dict, ...] ] 이라 [0]을 사용
        
        output_dict = {'ego': {}}

        cam_rgb_all_batch = []
        cam_to_ego_all_batch = []
        cam_intrinsic_all_batch = []

        gt_static_all_batch = []
        gt_dynamic_all_batch = []

        transformation_matrix_all_batch = []
        pairwise_t_matrix_all_batch = []
        # used to save each scenario's agent number
        record_len = []
        senario_id_all_batch = []
        agent_true_loc_all_batch = []
        cav_list_all_batch = []
        distance_all_batch = []
        single_bev_all_batch = []
        timestamp_all_batch = []
        
        for tick_dict in scenario_sequence:
            ego_dict = tick_dict['ego']
            
            camera_data = ego_dict['camera_data']
            camera_intrinsic = ego_dict['camera_intrinsic']
            camera_extrinsic = ego_dict['camera_extrinsic']

            assert camera_data.shape[0] == \
                   camera_intrinsic.shape[0] == \
                   camera_extrinsic.shape[0]

            record_len.append(camera_data.shape[0])

            cam_rgb_all_batch.append(camera_data)
            cam_intrinsic_all_batch.append(camera_intrinsic)
            cam_to_ego_all_batch.append(camera_extrinsic)

            # ground truth
            gt_dynamic_all_batch.append(ego_dict['gt_dynamic'])
            gt_static_all_batch.append(ego_dict['gt_static'])

            # transformation matrix
            transformation_matrix_all_batch.append(
                ego_dict['transformation_matrix'])
            senario_id_all_batch.append(ego_dict['scenario_id'][0])
            agent_true_loc_all_batch.append(ego_dict['agent_true_loc'])
            cav_list_all_batch.append(ego_dict['cav_list'])
            # distance_all_batch.append(ego_dict['dist_to_ego'])
            single_bev_all_batch.append(ego_dict['single_bev_imgae'])
            timestamp_all_batch.append(ego_dict['timestamp_key'])
            # pairwise matrix
            pairwise_t_matrix_all_batch.append(ego_dict['pairwise_t_matrix'])
            
        # (num_ticks, L, 1, M, H, W, C)
        cam_rgb_all_batch = torch.from_numpy(
            np.stack(cam_rgb_all_batch, axis=0)).unsqueeze(2).float()
        cam_intrinsic_all_batch = torch.from_numpy(
            np.stack(cam_intrinsic_all_batch, axis=0)).unsqueeze(2).float()
        cam_to_ego_all_batch = torch.from_numpy(
            np.stack(cam_to_ego_all_batch, axis=0)).unsqueeze(2).float()
        agent_true_loc_all_batch = torch.from_numpy(
            np.stack(agent_true_loc_all_batch, axis=0)).unsqueeze(2).float()
        single_bev_all_batch = torch.from_numpy(
            np.stack(single_bev_all_batch, axis=0)).unsqueeze(2).float()
        timestamp_all_batch = torch.from_numpy(
            np.stack(timestamp_all_batch, axis=0)).unsqueeze(2).float()
        
        # (num_ticks,)
        record_len = torch.from_numpy(np.array(record_len, dtype=int))

        # (num_ticks, 1, H, W)
        gt_static_all_batch = \
            torch.from_numpy(np.stack(gt_static_all_batch)).long()
        gt_dynamic_all_batch = \
            torch.from_numpy(np.stack(gt_dynamic_all_batch)).long()

        # (num_ticks,max_cav,4,4)
        transformation_matrix_all_batch = \
            torch.from_numpy(np.stack(transformation_matrix_all_batch)).float()
        pairwise_t_matrix_all_batch = \
            torch.from_numpy(np.stack(pairwise_t_matrix_all_batch)).float()
        senario_id_all_batch = \
            torch.from_numpy(np.stack(senario_id_all_batch)).float()
        # distance_all_batch = \
        #     torch.from_numpy(np.stack(distance_all_batch)).float()
        
        # convert numpy arrays to torch tensor
        output_dict['ego'].update({
            'inputs': cam_rgb_all_batch,
            'extrinsic': cam_to_ego_all_batch,
            'intrinsic': cam_intrinsic_all_batch,
            'gt_static': gt_static_all_batch,
            'gt_dynamic': gt_dynamic_all_batch,
            'transformation_matrix': transformation_matrix_all_batch,
            'pairwise_t_matrix': pairwise_t_matrix_all_batch,
            'record_len': record_len,
            'scenario_id': senario_id_all_batch,
            'agent_true_loc' : agent_true_loc_all_batch,
            'cav_list' : cav_list_all_batch,
            # 'dist_to_ego' : distance_all_batch,
            'single_bev' : single_bev_all_batch,
            'timestamp_key' : timestamp_all_batch
        })

        return output_dict

    def post_process(self, batch_dict, output_dict):
        output_dict = self.post_processor.post_process(batch_dict,
                                                       output_dict)

        return output_dict
