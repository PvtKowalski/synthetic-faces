import os
import bpy
import csv
import sys
import numpy as np
import bpy_extras
from mathutils import Euler
import json
import io
# append directiory of script location, gotta do it because script is executed from blender directory,
# not from its loacation
script_dir = "absolute/path/"
sys.path.append(script_dir)
from params import params
from pose_functions import *



param_num_expr = params['param_num_expr'] 
param_num_poses = params['param_num_poses']
param_alpha_start_x = params['param_alpha_start_x']
param_alpha_end_x = params['param_alpha_end_x']
param_x_steps = params['param_x_steps']
param_alpha_start_y = params['param_alpha_start_y']
param_alpha_end_y = params['param_alpha_end_y']
param_y_steps = params['param_y_steps']
x_delta = np.abs(param_alpha_end_x - param_alpha_start_x)/(param_x_steps - 1) if param_x_steps != 1 else 0
y_delta = np.abs(param_alpha_end_y - param_alpha_start_y)/(param_y_steps - 1) if param_y_steps != 1 else 0

param_object_colors_segmentation = params['param_object_colors_segmentation']
param_resolution_x = params['param_resolution_x']
param_resolution_y = params['param_resolution_y']
param_camera_lens = params['param_camera_lens']

param_path_models_segmentation = params['param_path_models_segmentation']
param_path_models_keypoints = params['param_path_models_keypoints']
param_dataset_name = params['param_dataset_name']
param_scene_path = params['param_scene_path']
param_save_to_folder = params['param_save_to_folder']
poses_keys = sorted(pose_dict.keys())
standing_pose_keys = sorted(standing_pose_dict.keys())
expression_keys = sorted(expression_dict.keys())
num_poses = len(pose_dict)
num_standing_poses = len(standing_pose_dict)
num_expressions = len(expression_dict)

r_eye = [1062, 1057,1073,1076,1070, 1068]
l_eye = [7768, 7765,7749,7754,7760, 7762]
l_eyebrow = [7006,6997,6996,6998,6999,7024,6983,6980,6979,6981,6982,6987,6984]
r_eyebrow = [255,225,224,222,223,233,212,207,206,204,205,208,209]
nose = [220,136,135,164,5054]
nose_below1 = [288, 286, 313, 317, 7076, 7055, 7057]
nose_below2 = [358, 314, 343, 7077, 7119]
outer_mouth = [402, 410,456,455,7215,7170, 7162, 7166,7241,492,486,406]
inner_mouth = [432, 472,474,7228, 7192, 7242,493,487]
aroundface = [5243,11852,11853,11841,11840,11838,11839,11843,11842,11848,11846,11845,11844,11854,11847,11851,11850,11849,7457,7617,10366,7619,7618,7616,7620,7621,732,919,918,914,916,917,3698,915,737,5240,5241,5242,5238,5246,5235,5236,5237,5239,5233,5234,5230,5229,5231,5232,5245,5244]
vertexList = r_eye+l_eye+r_eyebrow+l_eyebrow+nose+nose_below1+nose_below2+outer_mouth+inner_mouth+aroundface


def get_skeleton_data(model_name, scene, cam):
    bones = bpy.data.objects[model_name].pose.bones
    skeleton = {}
    for bone_name, bone in bones.items():
        head, tail = bone.head, bone.tail  # of type Vector
        cam_head = bpy_extras.object_utils.world_to_camera_view(scene, cam, head).to_tuple()
        cam_tail = bpy_extras.object_utils.world_to_camera_view(scene, cam, tail).to_tuple()
        skeleton[bone_name] = (cam_head, cam_tail)
    return skeleton


def create_folders_segmentation():
    # images, metadata, face_shapes, skeletons, object_segmentation, points
    folder_names = ['images', 'metadata', 'face_shapes', 'skeletons', 'object_segmentation', 'points']
    for fldr in folder_names:
        if not os.path.exists(os.path.join(param_save_to_folder, param_dataset_name, fldr)):
            os.makedirs(os.path.join(param_save_to_folder, param_dataset_name, fldr))


def keypoints_task():
    """Provided with folder of mhx2 models renders them with keypoints, facial segmentation, metaparameters."""
    # write metadata file header
    if not os.path.exists(os.path.join(param_save_to_folder, param_dataset_name, "metadata/metadata.csv")):
        with open(os.path.join(param_save_to_folder, param_dataset_name, "metadata/metadata.csv"), "w") as datafile:
            datafile.write("render_id,model_id,model_filename,race,gender,age,age_years")
            bone_eyeR = "eyeR_head_x,eyeR_head_y,eyeR_head_z,eyeR_tail_x,eyeR_tail_y,eyeR_tail_z,"
            bone_eyeL = "eyeL_head_x,eyeL_head_y,eyeL_head_z,eyeL_tail_x,eyeL_tail_y,eyeL_tail_z,"
            head_bone = "head_bone_head_x,head_bone_head_y,head_bone_head_z,head_bone_tail_x,head_bone_tail_y,head_bone_tail_z"
            datafile.write(bone_eyeR+bone_eyeL+head_bone+"\n")

    # save segmentation colors to json
    with open(os.path.join(param_save_to_folder, param_dataset_name, "object_segmentation/_colors.json"), 'w') as colors:
        json.dump(param_object_colors_segmentation, colors)

    models_path = os.listdir(param_path_models_segmentation)
    models_path = sorted([p for p in models_path if p.endswith(".mhx2")])
    cur = 0  # model counter

    limit_n = None # var for testing that limits by model count
    cnt = 0  # render counter

    model_meta = []
    with open(os.path.join(param_path_models_segmentation, "info.csv"), "r") as f:
        for ff in f:
            model_meta.append(ff)

    for p in models_path:
        bpy.ops.wm.open_mainfile(filepath=param_scene_path)

        model_meta_i = model_meta[cur + 1][:-1]  # -1 to drop \n at the and

        scene = bpy.context.scene
        render = scene.render
        render.resolution_x = param_resolution_x
        render.resolution_y = param_resolution_y
        camera_obj = bpy.data.objects['Camera']

        bpy.ops.import_scene.makehuman_mhx2(filepath=os.path.join(param_path_models_segmentation, p))
        human_name = p[:-5].capitalize()
        body = bpy.data.objects[human_name + ":Body"]

        for mat in bpy.data.materials.keys():
            emission = bpy.data.materials[mat].node_tree.nodes.new('ShaderNodeEmission')
            lightpath = bpy.data.materials[mat].node_tree.nodes.new('ShaderNodeLightPath')
            transparent = bpy.data.materials[mat].node_tree.nodes.new('ShaderNodeBsdfTransparent')
            mixshader = bpy.data.materials[mat].node_tree.nodes.new('ShaderNodeMixShader')
            mixshader.name = 'MixShaderCustom'
            output = bpy.data.materials[mat].node_tree.nodes['Material Output']
            # connect all but last
            bpy.data.materials[mat].node_tree.links.new(
                lightpath.outputs[0], mixshader.inputs[0]
            )
            bpy.data.materials[mat].node_tree.links.new(
                transparent.outputs[0], mixshader.inputs[1]
            )
            bpy.data.materials[mat].node_tree.links.new(
                emission.outputs[0], mixshader.inputs[2]
            )
            for keyword in param_object_colors_segmentation:
                if keyword in mat:  # if kw is present in material name
                    emission.inputs[0].default_value = param_object_colors_segmentation[keyword]  # set mask color

        for mat in bpy.data.materials.keys():
            nodes = bpy.data.materials[mat].node_tree.nodes
            glossy = nodes.get("Glossy BSDF")
            if (glossy is not None) and ('High-poly:Eye' not in mat):  # eyes stay glossy
                glossy.inputs[1].default_value = 0.75

        poses_rand = np.random.permutation(num_standing_poses)
        expressions_rand = np.random.permutation(num_expressions)

        face_shapes_keys = list(filter(lambda x: x.startswith("Mfa"), bpy.data.objects[human_name].keys()))

        for poses in range(param_num_poses):  # iterating over subsets
            for expressions in range(param_num_expr):
                # set pose and expression
                select_standing_pose(standing_pose_keys[poses_rand[poses]])
                select_expression(expression_keys[expressions_rand[expressions]])
                # get face shape data
                face_shapes = {shape_key: bpy.data.objects[human_name][shape_key] for shape_key in face_shapes_keys}
                # aim the camera
                the_bone = bpy.data.objects[human_name].pose.bones["special03"] #spine003 for body aim
                obj = bpy.data.objects[human_name]
                loc = obj.matrix_world * the_bone.matrix * the_bone.location
                empty = bpy.data.objects["Empty"]
                empty.location = loc
                sun = bpy.data.objects["Sun"]
                area_pivot = bpy.data.objects["AreaEmpty"]

                for i in range(param_x_steps):
                    for j in range(param_y_steps):
                        # emergency crutch
                        path_image = os.path.join(param_save_to_folder, param_dataset_name, "images/render_%d.png" % cnt)
                        if os.path.isfile(path_image):
                            print('\t!!! pts-file exist, skip ... [{}]'.format(path_image))
                            cnt += 1
                            continue

                        # camera rotation
                        empty.rotation_euler = Euler((param_alpha_start_x + i * x_delta, 0,
                                                      param_alpha_start_y + j * y_delta))


                        # save face_shapes
                        with open(os.path.join(param_save_to_folder, param_dataset_name, "face_shapes/face_shapes_%d.json" %
                                cnt), 'w') as face_shape_file:
                            json.dump(face_shapes, face_shape_file)
                        # some random lights
                        area_pivot.rotation_euler = Euler((np.random.rand() * 0.16 - 0.08,
                                                       np.random.rand() * 0.16 - 0.08,
                                                       np.random.rand() * 0.16 - 0.08))

                        # set original textures and anti-aliasing
                        scene.cycles.filter_width = 1.4
                        for mat in bpy.data.materials.keys():
                            bpy.data.materials[mat].node_tree.links.new(
                                bpy.data.materials[mat].node_tree.nodes['Mix Shader.001'].outputs[0],
                                bpy.data.materials[mat].node_tree.nodes['Material Output'].inputs[0]
                            )

                        bpy.context.scene.render.filepath = \
                            os.path.join(param_save_to_folder, param_dataset_name, "images/render_%d.png" % cnt)
                        bpy.ops.render.render(write_still=True)

                        # keypoints
                        nMesh = body.to_mesh(scene=scene, apply_modifiers=True, settings='PREVIEW')
                        co_2d = [bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, nMesh.vertices[v].co).to_tuple()
                            for v in vertexList]

                        with open(os.path.join(param_save_to_folder, param_dataset_name, "points/points_%d.csv" % cnt), "w") as csvfile:
                            a = csv.writer(csvfile, delimiter=",", lineterminator='\n')
                            a.writerows(co_2d)	


                        # to determine the direction of look I use skeleton bones
                        sp03 = bpy.data.objects[p[:-5].capitalize()].pose.bones["special03"]
                        eyeR = bpy.data.objects[p[:-5].capitalize()].pose.bones["eye.R"]
                        eyeL = bpy.data.objects[p[:-5].capitalize()].pose.bones["eye.L"]
                        sp03_h, sp03_t = sp03.head, sp03.tail
                        eyeR_h, eyeR_t = eyeR.head, eyeR.tail
                        eyeL_h, eyeL_t = eyeL.head, eyeL.tail
                        sp03_h = bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, sp03_h).to_tuple()
                        sp03_t = bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, sp03_t).to_tuple()
                        eyeR_h = bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, eyeR_h).to_tuple()
                        eyeR_t = bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, eyeR_t).to_tuple()
                        eyeL_h = bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, eyeL_h).to_tuple()
                        eyeL_t = bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, eyeL_t).to_tuple()
                        

                        # set mask textures
                        for mat in bpy.data.materials.keys():
                            bpy.data.materials[mat].node_tree.links.new(
                                bpy.data.materials[mat].node_tree.nodes['MixShaderCustom'].outputs[0],
                                bpy.data.materials[mat].node_tree.nodes['Material Output'].inputs[0]
                            )
                        # save skeleton data
                        skel = get_skeleton_data(human_name, scene, camera_obj)
                        with open(os.path.join(param_save_to_folder, param_dataset_name, "skeletons/skeleton_%d.json" %
                                cnt), 'w') as skelfile:
                            json.dump(skel, skelfile)

                        # reduce anti-aliasing and render again
                        scene.cycles.filter_width = 0.01
                        bpy.context.scene.render.filepath = \
                            os.path.join(param_save_to_folder, param_dataset_name, "object_segmentation/mask_%d.png" % cnt)
                        bpy.ops.render.render(write_still=True)

                        with open(os.path.join(param_save_to_folder, param_dataset_name, "metadata/metadata.csv"),
                                  "a") as datafile:
                            #datafile.write(str(cnt) + ',' + model_meta_i + "\n")
                            datafile.write(str(cnt) + ',' + model_meta_i + "," +
                                           ",".join([str(t) for t in eyeR_h]) + "," +
                                           ",".join([str(t) for t in eyeR_t]) + "," +
                                           ",".join([str(t) for t in eyeL_h]) + "," +
                                           ",".join([str(t) for t in eyeL_t]) + "," +
                                           ",".join([str(t) for t in sp03_h]) + "," +
                                           ",".join([str(t) for t in sp03_t]) + "\n")

                        cnt += 1

        cur += 1
        if limit_n is None:
            continue
        elif cur == limit_n:
            break

def segmentation_task():
    # write metadata file header
    if not os.path.exists(os.path.join(param_save_to_folder, param_dataset_name, "metadata/metadata.csv")):
        with open(os.path.join(param_save_to_folder, param_dataset_name, "metadata/metadata.csv"), "w") as datafile:
            datafile.write("render_id,model_id,model_filename,race,gender,age,age_years\n")

    # save segmentation colors to json
    with open(os.path.join(param_save_to_folder, param_dataset_name, "object_segmentation/_colors.json"), 'w') as colors:
        json.dump(param_object_colors_segmentation, colors)

    models_path = os.listdir(param_path_models_segmentation)
    models_path = sorted([p for p in models_path if p.endswith(".mhx2")])
    cur = 0  # model counter

    limit_n = None  # var for testing that limits by model count
    cnt = 0  # render counter

    model_meta = []
    with open(os.path.join(param_path_models_segmentation, "info.csv"), "r") as f:
        for ff in f:
            model_meta.append(ff)

    for p in models_path:
        bpy.ops.wm.open_mainfile(filepath=param_scene_path)

        model_meta_i = model_meta[cur + 1][:-1]  # -1 to drop \n at the and

        scene = bpy.context.scene
        render = scene.render
        render.resolution_x = param_resolution_x
        render.resolution_y = param_resolution_y
        camera_obj = bpy.data.objects['Camera']

        bpy.ops.import_scene.makehuman_mhx2(filepath=os.path.join(param_path_models_segmentation, p))
        human_name = p[:-5].capitalize()
        # body = bpy.data.objects[human_name + ":Body"]

        for mat in bpy.data.materials.keys():
            emission = bpy.data.materials[mat].node_tree.nodes.new('ShaderNodeEmission')
            lightpath = bpy.data.materials[mat].node_tree.nodes.new('ShaderNodeLightPath')
            transparent = bpy.data.materials[mat].node_tree.nodes.new('ShaderNodeBsdfTransparent')
            mixshader = bpy.data.materials[mat].node_tree.nodes.new('ShaderNodeMixShader')
            mixshader.name = 'MixShaderCustom'
            output = bpy.data.materials[mat].node_tree.nodes['Material Output']
            # connect all but last
            bpy.data.materials[mat].node_tree.links.new(
                lightpath.outputs[0], mixshader.inputs[0]
            )
            bpy.data.materials[mat].node_tree.links.new(
                transparent.outputs[0], mixshader.inputs[1]
            )
            bpy.data.materials[mat].node_tree.links.new(
                emission.outputs[0], mixshader.inputs[2]
            )
            for keyword in param_object_colors_segmentation:
                if keyword in mat:  # if kw is present in material name
                    emission.inputs[0].default_value = param_object_colors_segmentation[keyword]  # set mask color

        for mat in bpy.data.materials.keys():
            nodes = bpy.data.materials[mat].node_tree.nodes
            glossy = nodes.get("Glossy BSDF")
            if (glossy is not None) and ('High-poly:Eye' not in mat):  # eyes stay glossy
                glossy.inputs[1].default_value = 0.75

        poses_rand = np.random.permutation(num_poses)
        expressions_rand = np.random.permutation(num_expressions)

        face_shapes_keys = list(filter(lambda x: x.startswith("Mfa"), bpy.data.objects[human_name].keys()))

        for poses in range(param_num_poses):  # iterating over subsets
            for expressions in range(param_num_expr):
                # set pose and expression
                select_standing_pose(poses_keys[poses_rand[poses]])
                select_expression(expression_keys[expressions_rand[expressions]])
                # get face shape data
                face_shapes = {shape_key: bpy.data.objects[human_name][shape_key] for shape_key in face_shapes_keys}
                # aim the camera
                the_bone = bpy.data.objects[human_name].pose.bones["spine03"]
                obj = bpy.data.objects[human_name]
                loc = obj.matrix_world * the_bone.matrix * the_bone.location
                empty = bpy.data.objects["Empty"]
                empty.location = loc
                lights = bpy.data.objects["Lights"]

                for i in range(param_x_steps):
                    for j in range(param_y_steps):
                        # emergency crutch
                        path_image = os.path.join(param_save_to_folder, param_dataset_name, "images/render_%d.png" % cnt)
                        if os.path.isfile(path_image):
                            print('\t!!! pts-file exist, skip ... [{}]'.format(path_image))
                            cnt += 1
                            continue

                        # camera rotation
                        empty.rotation_euler = Euler((param_alpha_start_x + i * x_delta, 0,
                                                      param_alpha_start_y + j * y_delta))

                        # save skeleton data
                        skel = get_skeleton_data(human_name, scene, camera_obj)
                        with open(os.path.join(param_save_to_folder, param_dataset_name, "skeletons/skeleton_%d.json" %
                                cnt), 'w') as skelfile:
                            json.dump(skel, skelfile)
                        # save face_shapes
                        with open(os.path.join(param_save_to_folder, param_dataset_name, '/face_shapes/face_shapes_%d.json' %
                                cnt), 'w') as face_shape_file:
                            json.dump(face_shapes, face_shape_file)
                        # some random light shifts 0.08 max
                        lights.rotation_euler = Euler((np.random.rand() * 0.16 - 0.08,
                                                       np.random.rand() * 0.16 - 0.08,
                                                       np.random.rand() * 0.16 - 0.08))
                        bpy.data.lamps["Point.001"].node_tree.nodes["Emission"].inputs[1].default_value = 750
                        bpy.data.lamps["Point.001"].node_tree.nodes["Emission"].inputs[1].default_value += \
                            np.random.rand() * 150

                        # set original textures and anti-aliasing
                        scene.cycles.filter_width = 1.4
                        for mat in bpy.data.materials.keys():
                            bpy.data.materials[mat].node_tree.links.new(
                                bpy.data.materials[mat].node_tree.nodes['Mix Shader.001'].outputs[0],
                                bpy.data.materials[mat].node_tree.nodes['Material Output'].inputs[0]
                            )

                        bpy.context.scene.render.filepath = \
                            os.path.join(param_save_to_folder, param_dataset_name, "images/render_%d.png" % cnt)
                        bpy.ops.render.render(write_still=True)

                        # set mask textures
                        for mat in bpy.data.materials.keys():
                            bpy.data.materials[mat].node_tree.links.new(
                                bpy.data.materials[mat].node_tree.nodes['MixShaderCustom'].outputs[0],
                                bpy.data.materials[mat].node_tree.nodes['Material Output'].inputs[0]
                            )

                        # reduce anti-aliasing and render again
                        scene.cycles.filter_width = 0.01
                        bpy.context.scene.render.filepath = \
                            os.path.join(param_save_to_folder, param_dataset_name, "object_segmentation/mask_%d.png" % cnt)
                        bpy.ops.render.render(write_still=True)

                        with open(os.path.join(param_save_to_folder, param_dataset_name, "metadata/metadata.csv"),
                                  "a") as datafile:
                            datafile.write(str(cnt) + ',' + model_meta_i + "\n")

                        cnt += 1

        cur += 1
        if limit_n is None:
            continue
        elif cur == limit_n:
            break


if __name__ == '__main__':
    create_folders_segmentation()
    keypoints_task()
    #segmentation_task()
