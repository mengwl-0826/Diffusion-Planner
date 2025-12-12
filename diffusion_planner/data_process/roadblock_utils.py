import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Union, List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMapFactory
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject


def normalize_angle(angle: np.ndarray):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class BreadthFirstSearchRoadBlock:
    """
    A class that performs iterative breadth first search. The class operates on the roadblock graph.
    """

    def __init__(
        self, start_roadblock_id: int, map_api: Optional[AbstractMap], forward_search: str = True
    ):
        """
        Constructor of BreadthFirstSearchRoadBlock class
        :param start_roadblock_id: roadblock id where graph starts
        :param map_api: map class in nuPlan
        :param forward_search: whether to search in driving direction, defaults to True
        """
        self._map_api: Optional[AbstractMap] = map_api
         # 初始化搜索队列：用deque（双端队列，BFS标准数据结构），首元素是起始路障对象，None用于标记“当前深度结束”
        self._queue = deque([self.id_to_roadblock(start_roadblock_id), None])
        # 父节点字典：记录每个“路障ID+深度”对应的前驱路障（用于后续回溯构建路径）
        # 键格式：路障ID_深度（比如“101_3”表示深度3的101号路障），值：到达该路障的前一个路障
        self._parent: Dict[str, Optional[RoadBlockGraphEdgeMapObject]] = dict()
        self._forward_search = forward_search

        #  lazy loaded
        self._target_roadblock_ids: List[str] = None

    def search(
        self, target_roadblock_id: Union[str, List[str]], max_depth: int
    ) -> Tuple[List[RoadBlockGraphEdgeMapObject], bool]:
        """
        Apply BFS to find route to target roadblock.
        :param target_roadblock_id: id of target roadblock
        :param max_depth: maximum search depth
        :return: tuple of route and whether a path was found
        """

        if isinstance(target_roadblock_id, str):
            target_roadblock_id = [target_roadblock_id]
        self._target_roadblock_ids = target_roadblock_id

        start_edge = self._queue[0] # 起始路障对象

        # Initial search states
        path_found: bool = False    # 路径是否找到的标记
        end_edge: RoadBlockGraphEdgeMapObject = start_edge  # 最终找到的路障（默认起始路障）
        end_depth: int = 1  # 最终找到的路障深度（起始路障深度=1）
        depth: int = 1  # 当前搜索深度
        # 记录起始路障的父节点：起始路障没有前驱，父节点为None
        self._parent[start_edge.id + f"_{depth}"] = None

        while self._queue:
            current_edge = self._queue.popleft()

            # Early exit condition 提前退出条件：当前深度超过最大限制，终止搜索
            if self._check_end_condition(depth, max_depth):
                break

            # Depth tracking 深度跟踪：遇到None表示当前深度结束，进入下一层
            if current_edge is None:
                depth += 1
                self._queue.append(None)    # 为下一层添加深度分隔
                if self._queue[0] is None:  # 若队列下一个还是None，说明所有层都处理完，终止
                    break
                continue

            # Goal condition 目标检查：当前路障是否是目标路障，且深度未超限制
            if self._check_goal_condition(current_edge, depth, max_depth):
                end_edge = current_edge
                end_depth = depth
                path_found = True
                break
            # 获取当前路障的邻居（下一跳路障）：根据搜索方向选择出边或入边
            neighbors = (
                current_edge.outgoing_edges if self._forward_search else current_edge.incoming_edges
            )

            # Populate queue
            for next_edge in neighbors:
                # if next_edge.id in self._candidate_lane_edge_ids_old:
                self._queue.append(next_edge)
                self._parent[next_edge.id + f"_{depth + 1}"] = current_edge
                end_edge = next_edge    # 更新最终路障（即使没找到目标，也记录最后处理的路障）
                end_depth = depth + 1

        return self._construct_path(end_edge, end_depth), path_found

    def id_to_roadblock(self, id: str) -> RoadBlockGraphEdgeMapObject:
        """
        Retrieves roadblock from map-api based on id
        :param id: id of roadblock
        :return: roadblock class
        """
        block = self._map_api._get_roadblock(id)
        block = block or self._map_api._get_roadblock_connector(id)
        return block

    @staticmethod
    def _check_end_condition(depth: int, max_depth: int) -> bool:
        """
        Check if the search should end regardless if the goal condition is met.
        :param depth: The current depth to check.
        :param target_depth: The target depth to check against.
        :return: whether depth exceeds the target depth.
        """
        return depth > max_depth

    def _check_goal_condition(
        self,
        current_edge: RoadBlockGraphEdgeMapObject,
        depth: int,
        max_depth: int,
    ) -> bool:
        """
        Check if the current edge is at the target roadblock at the given depth.
        :param current_edge: edge to check.
        :param depth: current depth to check.
        :param max_depth: maximum depth the edge should be at.
        :return: True if the lane edge is contain the in the target roadblock. False, otherwise.
        """
        return current_edge.id in self._target_roadblock_ids and depth <= max_depth

    def _construct_path(
        self, end_edge: RoadBlockGraphEdgeMapObject, depth: int
    ) -> List[RoadBlockGraphEdgeMapObject]:
        """
        Constructs a path when goal was found.
        :param end_edge: The end edge to start back propagating back to the start edge.
        :param depth: The depth of the target edge.
        :return: The constructed path as a list of RoadBlockGraphEdgeMapObject
        """
        path = [end_edge]
        path_id = [end_edge.id]

        while self._parent[end_edge.id + f"_{depth}"] is not None:
            path.append(self._parent[end_edge.id + f"_{depth}"])
            path_id.append(path[-1].id)
            end_edge = self._parent[end_edge.id + f"_{depth}"]
            depth -= 1

        if self._forward_search:
            path.reverse()
            path_id.reverse()

        return (path, path_id)


def get_current_roadblock_candidates(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblocks_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    heading_error_thresh: float = np.pi / 4,
    displacement_error_thresh: float = 3,
) -> Tuple[RoadBlockGraphEdgeMapObject, List[RoadBlockGraphEdgeMapObject]]:
    """
    Determines a set of roadblock candidate where ego is located
    :param ego_state: class containing ego state
    :param map_api: map object
    :param route_roadblocks_dict: dictionary of on-route roadblocks
    :param heading_error_thresh: maximum heading error, defaults to np.pi/4
    :param displacement_error_thresh: maximum displacement, defaults to 3
    :return: tuple of most promising roadblock and other candidates
    
    返回值：(最可能的路障, 候选路障列表)（元组）
        第一元素：优先级最高的路障（距离最近、朝向最匹配，优先选导航路线上的）；
        第二元素：所有符合筛选条件的候选路障（供后续备用）。

    范围筛选：先找到自车周边 1 米内的路障（主路障 + 连接段），确保候选路障在自车附近；
    精准筛选：对每个候选路障的车道，计算 “自车到车道的距离” 和 “自车与车道的朝向偏差”，过滤超出阈值的候选；
    优先级排序：优先选择 “导航路线上” 的候选路障（贴合导航），再 fallback 到 “非路线但最接近” 的路障。

    """
    ego_pose: StateSE2 = ego_state.rear_axle
    roadblock_candidates = []
    # 2. 定义要查询的地图层：主路障（ROADBLOCK）和路障连接段（ROADBLOCK_CONNECTOR）
    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    # 3. 查询自车周边1米内的所有路障（主路障+连接段）
    roadblock_dict = map_api.get_proximal_map_objects(
        point=ego_pose.point, radius=1.0, layers=layers
    )
    # 合并两种路障类型，得到初始候选列表
    roadblock_candidates = (
        roadblock_dict[SemanticMapLayer.ROADBLOCK]
        + roadblock_dict[SemanticMapLayer.ROADBLOCK_CONNECTOR]
    )
    # 4. 边界处理：若1米内没找到路障，扩大范围找最近的路障
    if not roadblock_candidates:
        for layer in layers:
            roadblock_id_, distance = map_api.get_distance_to_nearest_map_object(
                point=ego_pose.point, layer=layer
            )
            roadblock = map_api.get_map_object(roadblock_id_, layer)

            if roadblock:
                roadblock_candidates.append(roadblock)
    # 存储“导航路线上的候选路障”及其距离误差
    on_route_candidates, on_route_candidate_displacement_errors = [], []
    # 存储“非导航路线的候选路障”及其距离误差
    candidates, candidate_displacement_errors = [], []

    roadblock_displacement_errors = []
    roadblock_heading_errors = []
    # 遍历每个初始候选路障，计算距离和朝向误差
    for idx, roadblock in enumerate(roadblock_candidates):
        # 初始化当前路障的最小距离误差和朝向误差（默认无穷大）
        lane_displacement_error, lane_heading_error = np.inf, np.inf
        # 遍历当前路障包含的所有车道（一个路障可能有多条车道）
        for lane in roadblock.interior_edges:
            # 获取车道的基准路径（离散化的点列表，比如每隔1米一个点）
            lane_discrete_path: List[StateSE2] = lane.baseline_path.discrete_path
            # 提取车道离散点的坐标（转为numpy数组，方便计算）
            lane_discrete_points = np.array(
                [state.point.array for state in lane_discrete_path], dtype=np.float64
            )
            # 计算自车到车道每个离散点的距离（欧氏距离）
            lane_state_distances = (
                (lane_discrete_points - ego_pose.point.array[None, ...]) ** 2.0
            ).sum(axis=-1) ** 0.5
            # 找到距离自车最近的离散点索引
            argmin = np.argmin(lane_state_distances)

            heading_error = np.abs(
                normalize_angle(lane_discrete_path[argmin].heading - ego_pose.heading)
            )
            displacement_error = lane_state_distances[argmin]

            if displacement_error < lane_displacement_error:
                lane_heading_error, lane_displacement_error = (
                    heading_error,
                    displacement_error,
                )

            if (
                heading_error < heading_error_thresh
                and displacement_error < displacement_error_thresh
            ):
                if roadblock.id in route_roadblocks_dict.keys():
                    on_route_candidates.append(roadblock)
                    on_route_candidate_displacement_errors.append(displacement_error)
                else:
                    candidates.append(roadblock)
                    candidate_displacement_errors.append(displacement_error)

        roadblock_displacement_errors.append(lane_displacement_error)
        roadblock_heading_errors.append(lane_heading_error)
    # 优先级1：优先选择“导航路线上”的候选路障（距离最近的最优）
    if on_route_candidates:  # prefer on-route roadblocks
        return (
            on_route_candidates[np.argmin(on_route_candidate_displacement_errors)],
            on_route_candidates,
        )
    # 优先级2：若无路线上的候选，选择“非路线但距离最近”的候选
    elif candidates:  # fallback to most promising candidate
        return candidates[np.argmin(candidate_displacement_errors)], candidates

    # otherwise, just find any close roadblock
    # 优先级3：若都无符合阈值的候选，直接返回所有候选中距离最近的
    return (
        roadblock_candidates[np.argmin(roadblock_displacement_errors)],
        roadblock_candidates,
    )


def route_roadblock_correction(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblock_ids: List[str],
    search_depth_backward: int = 15,
    search_depth_forward: int = 30,
) -> List[str]:
    """
    Applies several methods to correct route roadblocks.
    :param ego_state: class containing ego state
    :param map_api: map object
    :param route_roadblocks_dict: dictionary of on-route roadblocks
    :param search_depth_backward: depth of forward BFS search, defaults to 15
    :param search_depth_forward:  depth of backward BFS search, defaults to 30
    :return: list of roadblock id's of corrected route
    返回修正后的路障 ID 列表 route_roadblock_ids确保:
    自车初始位置所在路障在列表中（路线与自车对齐）；
    列表中相邻路障在地图上连通（无断裂）；
    路线无环路（避免重复经过同一组路障
    """
    # 1. 创建字典，存储路障ID与对应的路障对象（主路障或路障连接器）
    route_roadblock_dict = {}
    for id_ in route_roadblock_ids:
        block = map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
        block = block or map_api.get_map_object(
            id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
        )
        route_roadblock_dict[id_] = block
    # 3. 根据自车状态，找到当前所在/邻近的路障（候选列表）
    starting_block, starting_block_candidates = get_current_roadblock_candidates(
        ego_state, map_api, route_roadblock_dict
    )
    starting_block_ids = [roadblock.id for roadblock in starting_block_candidates]
    # 2. 提取路障对象列表和ID列表（方便后续处理）
    route_roadblocks = list(route_roadblock_dict.values())
    route_roadblock_ids = list(route_roadblock_dict.keys())

    # Fix 1: when agent starts off-route 修正 “自车初始位置偏离规划路线” 问题, （比如规划路线是 [B,C,D]，但自车在 A）
    if starting_block.id not in route_roadblock_ids:
        # Backward search if current roadblock not in route
        # 方案1：反向搜索（从规划路线起点往自车方向找）
        # 搜索目标：从规划路线第一个路障（route_roadblock_ids[0]）出发，反向找自车所在的路障(这里用的candidates)
        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[0], map_api, forward_search=False
        )
        (path, path_id), path_found = graph_search.search(
            starting_block_ids, max_depth=search_depth_backward
        )

        if path_found:
            # [:0]：在列表最前面插入
            # path[:-1]：截取搜索路径的 “除最后一个元素外的所有元素
            route_roadblocks[:0] = path[:-1]
            route_roadblock_ids[:0] = path_id[:-1]

        else:
            # 反向没找到，执行方案2：正向搜索（从自车路障往规划路线找）
            # 搜索目标：从自车路障（starting_block.id）出发，正向找规划路线的前3个路障
            # Forward search to any route roadblock
            graph_search = BreadthFirstSearchRoadBlock(
                starting_block.id, map_api, forward_search=True
            )
            (path, path_id), path_found = graph_search.search(
                route_roadblock_ids[:3], max_depth=search_depth_forward
            )

            if path_found:# 找到连接路径（比如自车A→B，规划路线是[B,C,D]）
                # 找到连接路径的终点（B）在规划路线中的位置
                end_roadblock_idx = np.argmax(
                    np.array(route_roadblock_ids) == path_id[-1]
                )
                # 裁剪规划路线：去掉B之前的无效段（这里B是起点，裁剪后还是[B,C,D]）
                route_roadblocks = route_roadblocks[end_roadblock_idx + 1 :]
                route_roadblock_ids = route_roadblock_ids[end_roadblock_idx + 1 :]
                # 把自车到B的路径（A→B）插入到规划路线最前面，变成[A,B,C,D]
                route_roadblocks[:0] = path
                route_roadblock_ids[:0] = path_id

    # Fix 2: check if roadblocks are linked, search for links if not 修正 “规划路线中路障不连通” 问题
    # 比如规划路线是 [A,C]，但 A 和 C 之间没有通路，需要补全中间的连接路障 B
    roadblocks_to_append = {}
    # 遍历规划路线中相邻的路障对（i和i+1）
    for i in range(len(route_roadblocks) - 1):
        # 获取第i+1个路障的“入边路障ID”（即能直接连接到i+1的路障）
        next_incoming_block_ids = [
            _roadblock.id for _roadblock in route_roadblocks[i + 1].incoming_edges
        ]
        # 判断第i个路障是否能直接连接到第i+1个路障
        is_incoming = route_roadblock_ids[i] in next_incoming_block_ids

        if is_incoming:
            continue
        # 不能直接连接，用BFS搜索i和i+1之间的连通路径
        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[i], map_api, forward_search=True
        )
        (path, path_id), path_found = graph_search.search(
            route_roadblock_ids[i + 1], max_depth=search_depth_forward
        )
        # 找到有效路径（且路径长度≥3，避免过短的无效连接）
        if path_found and path and len(path) >= 3:
            # 裁剪路径：去掉首尾（避免重复i和i+1），只保留中间连接段
            path, path_id = path[1:-1], path_id[1:-1]
            # 记录插入位置和中间路障
            roadblocks_to_append[i] = (path, path_id)

    # append missing intermediate roadblocks
    offset = 1 # 插入后路线长度变化，需要偏移量修正
    # 在i和i+1之间插入中间路障
    for i, (path, path_id) in roadblocks_to_append.items():
        route_roadblocks[i + offset : i + offset] = path
        route_roadblock_ids[i + offset : i + offset] = path_id
        offset += len(path)

    # Fix 3: cut route-loops 修正 “规划路线存在环路” 问题
    # 规划路线中出现重复路障（比如 [A,B,C,B,D]），导致路线循环，需要移除环路，保留唯一路径
    route_roadblocks, route_roadblock_ids = remove_route_loops(
        route_roadblocks, route_roadblock_ids
    )

    return route_roadblock_ids


def remove_route_loops(
    route_roadblocks: List[RoadBlockGraphEdgeMapObject],
    route_roadblock_ids: List[str],
) -> Tuple[List[str], List[RoadBlockGraphEdgeMapObject]]:
    """
    Remove ending of route, if the roadblock are intersecting the route (forming a loop).
    :param route_roadblocks: input route roadblocks
    :param route_roadblock_ids: input route roadblocks ids
    :return: tuple of ids and roadblocks of route without loops
    """

    roadblock_occupancy_map = None
    loop_idx = None

    for idx, roadblock in enumerate(route_roadblocks):
        # loops only occur at intersection, thus searching for roadblock-connectors.
        if str(roadblock.__class__.__name__) == "NuPlanRoadBlockConnector":
            if not roadblock_occupancy_map:
                roadblock_occupancy_map = STRTreeOccupancyMapFactory.get_from_geometry(
                    [roadblock.polygon], [roadblock.id]
                )
                continue

            strtree, index_by_id = roadblock_occupancy_map._build_strtree()
            indices = strtree.query(roadblock.polygon)
            if len(indices) > 0:
                for geom in strtree.geometries.take(indices):
                    area = geom.intersection(roadblock.polygon).area
                    if area > 1:
                        loop_idx = idx
                        break
                if loop_idx:
                    break

            roadblock_occupancy_map.insert(roadblock.id, roadblock.polygon)

    if loop_idx:
        route_roadblocks = route_roadblocks[:loop_idx]
        route_roadblock_ids = route_roadblock_ids[:loop_idx]

    return route_roadblocks, route_roadblock_ids
