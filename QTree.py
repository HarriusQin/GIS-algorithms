from __future__ import annotations
import json
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class QTree:
    # MEMBERS
    # Spatial Range
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    # Whether this node has been divided
    divided: bool
    # Capacity of one undivided node
    capacity: int
    # Minimal size of a node
    size_min: float
    # Children of this node, which do not exist before this node divided
    nw: QTree | None
    ne: QTree | None
    sw: QTree | None
    se: QTree | None
    # Data: [(x,y,idx)], idx refer to the index of a row in the table/list/array of points
    points: list[tuple[float, float, int]]

    # Definitions of coordinates
    #    ---------  <- y_max
    #   | nw | ne |
    #   |----|----|
    #   | sw | se |
    #    ----|----  <- y_min
    #   ^         ^
    #  x_min    x_max

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        capacity: int,
        size_min: float = 0,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.divided = False
        self.capacity = capacity
        self.size_min = size_min
        self.points = []
        self.nw = self.ne = self.sw = self.se = None

    def insert(self, x: float, y: float, idx: int) -> bool:
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return False
        if self.divided:
            return (
                self.nw.insert(x, y, idx)
                or self.ne.insert(x, y, idx)
                or self.sw.insert(x, y, idx)
                or self.se.insert(x, y, idx)
            )
        self.points.append((x, y, idx))
        if len(self.points) > self.capacity:
            self._subdivide()
        return True

    def _subdivide(self):
        if (self.x_max - self.x_min) / 2 < self.size_min or (
            self.y_max - self.y_min
        ) / 2 < self.size_min:
            return
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2
        self.nw = QTree(
            self.x_min, x_mid, y_mid, self.y_max, self.capacity, self.size_min
        )
        self.ne = QTree(
            x_mid, self.x_max, y_mid, self.y_max, self.capacity, self.size_min
        )
        self.sw = QTree(
            self.x_min, x_mid, self.y_min, y_mid, self.capacity, self.size_min
        )
        self.se = QTree(
            x_mid, self.x_max, self.y_min, y_mid, self.capacity, self.size_min
        )
        for point in self.points:
            self.nw.insert(point[0], point[1], point[2]) or self.ne.insert(
                point[0], point[1], point[2]
            ) or self.sw.insert(point[0], point[1], point[2]) or self.se.insert(
                point[0], point[1], point[2]
            )
        self.points = []
        self.divided = True

    def _intersect(
        self, min_x: float, max_x: float, min_y: float, max_y: float
    ) -> bool:
        return not (
            max_x < self.x_min
            or min_x > self.x_max
            or max_y < self.y_min
            or min_y > self.y_max
        )

    def query_rect(
        self, min_x: float, max_x: float, min_y: float, max_y: float
    ) -> list[tuple[float, float, int]]:
        found = []
        if not self._intersect(min_x, max_x, min_y, max_y):
            return found
        if self.divided:
            found += self.nw.query_rect(min_x, max_x, min_y, max_y)
            found += self.ne.query_rect(min_x, max_x, min_y, max_y)
            found += self.sw.query_rect(min_x, max_x, min_y, max_y)
            found += self.se.query_rect(min_x, max_x, min_y, max_y)
        else:
            for point in self.points:
                if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                    found.append(point)
        return found

    def to_dict(self) -> dict:
        """
        将QTree转化为字典
        :return: 转化得到的字典
        """
        table = {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "capacity": self.capacity,
            "size_min": self.size_min,
            "divided": self.divided,
            "points": self.points,
        }
        if self.divided:
            table["nw"] = self.nw.to_dict()
            table["ne"] = self.ne.to_dict()
            table["sw"] = self.sw.to_dict()
            table["se"] = self.se.to_dict()
        return table

    @classmethod
    def from_dict(cls, table: dict) -> QTree:
        """
        从字典构造QTree
        :param table: 具有以下格式的字典——{
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "capacity": capacity, "size_min": size_min,
            "divided": divided, "points": points,
            [Optional] "nw" : dict, "ne" : dict, "sw" : dict, "se" : dict
        }
        转化为四叉树，这个字典可以用to_dict方法创建或是用load_tree_from_file函数从json文件读取
        :return: 转化得到的QTree
        """
        qt = cls(
            table["x_min"],
            table["x_max"],
            table["y_min"],
            table["y_max"],
            capacity=table["capacity"],
            size_min=table["size_min"],
        )
        qt.points = table["points"]
        qt.divided = table["divided"]
        if table["divided"]:
            qt.nw = cls.from_dict(table["nw"])
            qt.ne = cls.from_dict(table["ne"])
            qt.sw = cls.from_dict(table["sw"])
            qt.se = cls.from_dict(table["se"])
        return qt

    def visualize(
        self,
        ax: Optional[object] = None,
        show_points: bool = True,
        show_rects: bool = True,
        edgecolor: str = "gray",
        pointcolor: str = "blue",
        linewidth: float = 0.5,
    ):
        """

        :param ax: matplotlib axes 对象，若为 None 则新建一个
        :param show_points: 是否绘制点，bool类型
        :param show_rects: 是否绘制四叉树的矩形边界，bool类型
        :param edgecolor: 矩形边框颜色，目前可以使用str类型
        :param pointcolor: 点的颜色，目前可以使用str类型
        :param linewidth: 线框的宽度，float类型
        :return: 无
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect("equal")
            created_ax = True
        else:
            created_ax = False

        # 绘制当前节点的边界框
        if show_rects:
            rect = Rectangle(
                (self.x_min, self.y_min),
                self.x_max - self.x_min,
                self.y_max - self.y_min,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor="none",
            )
            ax.add_patch(rect)

        # 如果是叶子节点且有点，绘制点
        if not self.divided and show_points and self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            ax.scatter(xs, ys, color=pointcolor, s=10)

        # 递归绘制子节点
        if self.divided:
            self.nw.visualize(
                ax, show_points, show_rects, edgecolor, pointcolor, linewidth
            )
            self.ne.visualize(
                ax, show_points, show_rects, edgecolor, pointcolor, linewidth
            )
            self.sw.visualize(
                ax, show_points, show_rects, edgecolor, pointcolor, linewidth
            )
            self.se.visualize(
                ax, show_points, show_rects, edgecolor, pointcolor, linewidth
            )

        if created_ax:
            plt.show()


def save_tree_to_file(
    tree: QTree,
    filename: str,
) -> None:
    with open(filename, "w") as file:
        json.dump(tree.to_dict(), file)


def load_tree_from_file(filename: str) -> QTree:
    with open(filename, "r") as file:
        return QTree.from_dict(json.load(file))



