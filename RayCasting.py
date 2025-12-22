from typing import Union

from shapely.geometry import Polygon, LineString, LinearRing, Point, MultiPolygon


def point_on_segment(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float, eps: float = 0
) -> bool:
    """
    检查点P(px,py)是否在边{(x1,y1),(x2,y2)}上
    :param px: 点P的x坐标，float类型
    :param py: 点P的y坐标，float类型
    :param x1: 边的端点1的x坐标，float类型
    :param y1: 边的端点1的y坐标，float类型
    :param x2: 边的端点2的x坐标，float类型
    :param y2: 边的端点2的y坐标，float类型
    :param eps: 容差值，float类型
    :return: 点P是否在边上，bool类型
    """
    if not (
        min(x1, x2) - eps <= px <= max(x1, x2) + eps
        and min(y1, y2) - eps <= py <= max(y1, y2) + eps
    ):
        return False

    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False

    return True


def in_ring_with_boundary(
    px: float, py: float, ring_coords: list[tuple[float, float]], eps: float = 0
) -> bool:
    """
    射线交叉法——
    利用从待测点水平向右的射线，检查点P(px,py)是否在坐标序列ring_coords所定义的环当中
    :param px: 点P的x坐标，float类型
    :param py: 点P的y坐标，float类型
    :param ring_coords: 定义一个环的坐标序列[(x1,y1),...,(xn,yn)]，list类型
    :param eps: 容差值，float类型
    :return: 点P是否在坐标序列ring_coords所定义的环当中，float类型
    """
    n = len(ring_coords)
    if n < 3:
        return False

    # Step 1: 检查是否与顶点重合
    for x, y in ring_coords:
        if abs(px - x) <= eps and abs(py - y) <= eps:
            return True

    # Step 2: 检查是否在任意边上
    for i in range(n):
        x1, y1 = ring_coords[i]
        x2, y2 = ring_coords[(i + 1) % n]
        if point_on_segment(px, py, x1, y1, x2, y2, eps):
            return True

    # Step 3: 射线法判断严格内部
    inside = False
    x_old, y_old = ring_coords[0]

    for i in range(1, n + 1):
        x_new, y_new = ring_coords[i % n]

        # 跳过水平边
        if abs(y_new - y_old) <= eps:
            pass
        else:
            # 确保 y_old <= y_new
            if y_old > y_new:
                x_low, y_low = x_new, y_new
                x_high, y_high = x_old, y_old
            else:
                x_low, y_low = x_old, y_old
                x_high, y_high = x_new, y_new

            if y_low - eps <= py < y_high - eps:  # 注意：上边界开区间，避免重复计数
                # 计算射线与边的交点 x 坐标
                x_intersect = (x_high - x_low) * (py - y_low) / (y_high - y_low) + x_low
                if px < x_intersect - eps:
                    inside = not inside

        x_old, y_old = x_new, y_new

    return inside


def point_in_polygon(point: Point, polygon: Union[Polygon, MultiPolygon]) -> bool:
    """
    判断点是否在多边形内（支持 Polygon 和 MultiPolygon）
    :param point: 需要判断的点，shapely.geometry.Point类型
    :param polygon: 多边形，shapely.geometry.Polygon or MultiPolygon类型
    :return: 点是否位于多边形对象当中，bool类型
    """
    if isinstance(polygon, MultiPolygon):
        # MultiPolygon：只要在一个子多边形内就算在内
        return any(point_in_simple_polygon(point, poly) for poly in polygon.geoms)
    elif isinstance(polygon, Polygon):
        return point_in_simple_polygon(point, polygon)
    else:
        raise ValueError("Geometry must be Polygon or MultiPolygon")


def point_in_simple_polygon(point: Point, poly: Polygon) -> bool:
    """
    正对简单的单个多边形进行判断
    :param point: 需要判断的点，shapely.geometry.Point类型
    :param poly: 单个多边形对象，shapely.geometry.Polygon类型
    :return: 点是否位于多边形对象当中，bool类型
    """
    x, y = point.x, point.y
    # 先判断是否在外接矩形内
    minx, miny, maxx, maxy = poly.bounds
    if not (minx <= x <= maxx and miny <= y <= maxy):
        return False

    # 处理外环（exterior）
    if not in_ring_with_boundary(x, y, poly.exterior.coords):
        return False

    # 处理内环（interiors）：如果在任一洞内，则不在多边形内
    for interior in poly.interiors:
        if in_ring_with_boundary(x, y, interior.coords):
            return False

    return True

