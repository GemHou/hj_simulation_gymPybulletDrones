import pybullet as p


def vis_point(point, color=None):
    if color is None:
        color = [1, 0, 1, 0.5]

    # 创建可视化描述符（球体）
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.1,
        rgbaColor=color
    )
    # 创建多体对象
    p.createMultiBody(
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=point,
        baseOrientation=[0, 0, 0, 1],
        baseMass=0
    )


def visualize_path(path_points_pos):
    for point in path_points_pos:
        vis_point(point)
