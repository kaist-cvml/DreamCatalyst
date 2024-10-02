import pathlib
import pydiffvg
import torch
import numpy as np

gamma = 1.0

def get_path(pos_init_method):
    num_segments = 4 # of 5

    points = []
    # if self.segment_init == 'circle':
    num_control_points = [2] * num_segments
    radius = 20
    center = pos_init_method()
    bias = center
    # color_ref = copy.deepcopy(bias)

    avg_degree = 360 / (num_segments * 3)
    for i in range(0, num_segments * 3):
        point = (
            np.cos(np.deg2rad(i * avg_degree)), np.sin(np.deg2rad(i * avg_degree))
        )
        points.append(point)

    points = torch.FloatTensor(points) * radius + torch.FloatTensor(bias).unsqueeze(dim=0)

    path = pydiffvg.Path(
        num_control_points=torch.LongTensor(num_control_points),
        points=points,
        stroke_width=torch.tensor(0.0),
        is_closed=True
    )
    
    return path

def set_points_parameters(cur_shapes, id_delta=0):
    # stroke`s location optimization
    for i, path in enumerate(cur_shapes):
        path.id = i + id_delta  # set point id
        path.points.requires_grad = True

def reinitialize_paths(
                    # reinit_path: bool = False,
                    opacity_threshold: float = None,
                    area_threshold: float = None,
                    fpath: pathlib.Path = None,
                    cur_shape_groups = None,
                    cur_shapes = None,
                    canvas_width = None,
                    canvas_height = None,
                    pos_init_method = None,
                    step = None
                    ):
    """
    reinitialize paths, also known as 'Reinitializing paths' in VectorFusion paper.

    Args:
        reinit_path: whether to reinitialize paths or not.
        opacity_threshold: Threshold of opacity.
        area_threshold: Threshold of the closed polygon area.
        fpath: The path to save the reinitialized SVG.
    """
    # re-init by opacity_threshold
    select_path_ids_by_opc = []
    if opacity_threshold != 0 and opacity_threshold is not None:
        def get_keys_below_threshold(my_dict, threshold):
            keys_below_threshold = [key for key, value in my_dict.items() if value < threshold]
            return keys_below_threshold

        opacity_record_ = {group.shape_ids.item(): group.fill_color.data[-1].item()
                            for group in cur_shape_groups}
        # print("-> opacity_record: ", opacity_record_)
        print("-> opacity_record: ", [f"{k}: {v:.3f}" for k, v in opacity_record_.items()])
        select_path_ids_by_opc = get_keys_below_threshold(opacity_record_, opacity_threshold)
        print("select_path_ids_by_opc: ", select_path_ids_by_opc)

    # remove path by area_threshold
    select_path_ids_by_area = []
    if area_threshold != 0 and area_threshold is not None:
        area_records = [Polygon(shape.points.detach().numpy()).area for shape in cur_shapes]
        # print("-> area_records: ", area_records)
        print("-> area_records: ", ['%.2f' % i for i in area_records])
        for i, shape in enumerate(cur_shapes):
            if Polygon(shape.points.detach().numpy()).area < area_threshold:
                select_path_ids_by_area.append(shape.id)
        print("select_path_ids_by_area: ", select_path_ids_by_area)


    # re-init paths
    reinit_union = list(set(select_path_ids_by_opc + select_path_ids_by_area))
    if len(reinit_union) > 0:
        # print(reinit_union)
        new_point_vars = []
        for i, path in enumerate(cur_shapes):
            if path.id in reinit_union:
                # print(path.id)
                # print(f's : {i}')
                new_path = get_path(pos_init_method)
                cur_shapes[i] = new_path
                new_path.points.requires_grad = True
                new_point_vars.append(new_path.points)

        new_color_vars = []
        for i, group in enumerate(cur_shape_groups):
            shp_ids = group.shape_ids.cpu().numpy().tolist()
            if set(shp_ids).issubset(reinit_union):
                # print(f'sg : {i}')
                # print(shp_ids)
                fill_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
                fill_color_init[-1] = np.random.uniform(0.7, 1)
                stroke_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
                cur_shape_groups[i] = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor(list(shp_ids)),
                    fill_color=fill_color_init,
                    stroke_color=stroke_color_init)
                fill_color_init.requires_grad = True
                new_color_vars.append(fill_color_init)

        # save reinit svg
        pydiffvg.save_svg(f'{fpath}/reinit_{step}.svg',
                              canvas_width, canvas_height, cur_shapes, cur_shape_groups)

        print("-" * 40)
    return cur_shapes, cur_shape_groups, new_point_vars, new_color_vars


def render(canvas_width, canvas_height, shapes, shape_groups, base_path, save_file):
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)

    img = pydiffvg.RenderFunction.apply(canvas_width, # width
                canvas_height, # height
                2,   # num_samples_x
                2,   # num_samples_y
                0,   # seed
                None, # bg
                *scene_args)
    
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])

    # The output image is in linear RGB space. Do Gamma correction before saving the image.
    pydiffvg.imwrite(img.cpu(), f'{base_path}/{save_file}.png', gamma=gamma)

    return img

def clip_curve_shape(shape_groups, train_stroke=False):
    for group in shape_groups:
        if train_stroke:
            group.stroke_color.data.clamp_(0.0, 1.0)
        else:
            group.fill_color.data.clamp_(0.0, 1.0)
