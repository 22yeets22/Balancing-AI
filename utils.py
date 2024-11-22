from constants import HEIGHT


def convert_coord(point):
    return int(point[0]), int(HEIGHT - point[1])


def convert_shape(shape, body):
    vertices = shape.get_vertices()
    vertices = [(v.rotated(body.angle) + body.position) for v in vertices]
    vertices = [convert_coord(v) for v in vertices]
    return vertices


def linear_conv(point, point_min, point_max, transform_min, transform_max):
    return (point - point_min) / (point_max - point_min) * (transform_max - transform_min) + transform_min
