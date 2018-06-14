import numpy as np

params = {
	'param_num_expr': 2,
	'param_num_poses': 1,
	# camera rotation, x is rotation around horizontal axis, y - around vertical
	'param_alpha_start_x': 0, # -np.pi / 8
	'param_alpha_end_x': 0, # np.pi / 8
	'param_x_steps': 1,
	'param_alpha_start_y': -np.pi / 6,
	'param_alpha_end_y': np.pi / 6.2,
	'param_y_steps': 3,
	'param_light_strength': None,  # todo
	'param_get_depth': True, # todo
	'param_get_normals': False,
	'param_get_skeleton': True,
	'param_get_metadata': True,
	'param_get_segmentation': True,
	'param_object_colors_segmentation': {
		'Body': (1, 0, 0, 1),  # RGBA tuples for object segmentation
		'Eyebrow': (0.5, 0.25, 0, 1),  #
		'Eyelashe': (1, 0, 0, 1),  # (0.5, 0, 0.25, 1) segmenting eyelashes disabled
		'High-poly:Eye': (0, 0, 1, 1),  #
		'suit': (0.25, 0.5, 0, 1),  #
		'Shoe': (0.5, 0.1, 0.1, 1),  #
		'Short': (0.8, 0.8, 0, 1),  # hair !
		'Afro': (0.8, 0.8, 0, 1),
		'Bob': (0.8, 0.8, 0, 1),
		'Braid': (0.8, 0.8, 0, 1),
		'Long': (0.8, 0.8, 0, 1),
		'Ponytail': (0.8, 0.8, 0, 1),
		'Teeth': (0.8, 0.8, 0.8, 1),  #
		'Tongue': (0.25, 0.25, 0.75, 1)  #
	},
	'param_resolution_x': 256,
	'param_resolution_y': 256,
	'param_camera_lens': 60.0,
	'param_camera_distance': None,
	# use absolute paths here because blender scripting works in mysterious ways
	'param_path_models_segmentation': "set/absolute/path/models-nocloth-256",
	# keypoints mesh indices are not preserved if base mesh is altered with clothes so use naked models for keypoints
	'param_path_models_keypoints': "set/absolute/path/models-nocloth-256",
	'param_dataset_name': "dataset_new",
	'param_scene_path': "absolute/path/CleanSceneNew.blend",
	'param_save_to_folder': "absolute/path/"
}