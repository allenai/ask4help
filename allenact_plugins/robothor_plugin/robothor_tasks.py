import math
from typing import Tuple, List, Dict, Any, Optional, Union, Sequence, cast

import gym
import numpy as np

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.system import get_logger
from allenact.utils.tensor_utils import tile_images
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.robothor_plugin.robothor_constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    END,
    LOOK_UP,
    LOOK_DOWN,
)
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment


def spl_metric(
    success: bool, optimal_distance: float, travelled_distance: float
) -> Optional[float]:
    if not success:
        return 0.0
    elif optimal_distance < 0:
        return None
    elif optimal_distance == 0:
        if travelled_distance == 0:
            return 1.0
        else:
            return 0.0
    else:
        travelled_distance = max(travelled_distance, optimal_distance)
        return optimal_distance / travelled_distance


class PointNavTask(Task[RoboThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END)

    def __init__(
        self,
        env: RoboThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_configs: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.last_geodesic_distance = self.env.distance_to_point(
            self.task_info["target"]
        )

        self.optimal_distance = self.last_geodesic_distance
        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List[
            Any
        ] = []  # the initial coordinate will be directly taken from the optimal path
        self.travelled_distance = 0.0

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["action_names"] = self.action_names()

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.class_action_names()[action]

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
        if len(self.path) > 1:
            self.travelled_distance += IThorEnvironment.position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> Optional[bool]:
        tget = self.task_info["target"]
        dist = self.dist_to_target()

        if -0.5 < dist <= 0.2:
            return True
        elif dist > 0.2:
            return False
        else:
            get_logger().debug(
                "No path for {} from {} to {}".format(
                    self.env.scene_name, self.env.agent_state(), tget
                )
            )
            return None

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.dist_to_target()

        if geodesic_distance == -1.0:
            geodesic_distance = self.last_geodesic_distance
        if (
            self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5
        ):  # (robothor limits)
            rew += self.last_geodesic_distance - geodesic_distance
        self.last_geodesic_distance = geodesic_distance

        return rew * self.reward_configs["shaping_weight"]

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()

        if self._took_end_action:
            if self._success is not None:
                reward += (
                    self.reward_configs["goal_success_reward"]
                    if self._success
                    else self.reward_configs["failed_stop_reward"]
                )
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_configs.get("reached_max_steps_reward", 0.0)

        self._rewards.append(float(reward))
        return float(reward)

    def dist_to_target(self):
        return self.env.distance_to_point(self.task_info["target"])

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        total_reward = float(np.sum(self._rewards))
        self._rewards = []

        if self._success is None:
            return {}

        dist2tget = self.dist_to_target()
        spl = spl_metric(
            success=self._success,
            optimal_distance=self.optimal_distance,
            travelled_distance=self.travelled_distance,
        )

        metrics = {
            **super(PointNavTask, self).metrics(),
            "success": self._success,  # False also if no path to target
            "total_reward": total_reward,
            "dist_to_target": dist2tget,
            "spl": 0 if spl is None else spl,
        }
        return metrics


class ObjectNavTask(Task[RoboThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END, LOOK_UP, LOOK_DOWN)

    def __init__(
        self,
        env: RoboThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_configs: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.mirror = task_info["mirrored"]

        self._all_metadata_available = env.all_metadata_available

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List = (
            []
        )  # the initial coordinate will be directly taken from the optimal path
        self.travelled_distance = 0.0

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["taken_actions"] = []
        self.task_info["action_names"] = self.class_action_names()
        self.task_info["taken_ask_actions"] = []

        if self._all_metadata_available:
            self.last_geodesic_distance = self.env.distance_to_object_type(
                self.task_info["object_type"]
            )
            self.optimal_distance = self.last_geodesic_distance
            self.closest_geo_distance = self.last_geodesic_distance

        self.last_expert_action: Optional[int] = None
        self.agent_asked_for_help = False
        self.num_steps_expert = 0
        self.help_asked_at_all = False
        self.false_stop = 0
        self.asked_init_help_flag = False
        self.expert_action_span = 0
        self.max_expert_span = 0
        self.expert_ends_traj = False
        self.expert_took_step = False 

        self.penalty_given_once = False

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:

        ask_action = action['ask_action']
        ask_action = cast(int,ask_action)
        
        if ask_action==0:
            # print ('expert takes step')
            ask_action_str = 'start_asking'
            self.agent_asked_for_help = True 
            self.help_asked_at_all = True
            self.expert_action_span+=1
            self.max_expert_span = max(self.expert_action_span,self.max_expert_span)

        if ask_action==1:
            # print ('agent takes step')  
            ask_action_str = 'stop_asking'  
            self.agent_asked_for_help = False 
            # self.max_expert_span = max(self.expert_action_span,self.max_expert_span)
            self.expert_action_span = 0 ##reset counter  


        '''    
        if ask_action==1:
            # print ('start asking for help')
            self.agent_asked_for_help = True
            self.help_asked_at_all = True
            self.expert_action_span+=1
            self.asked_init_help_flag = False
            # self.max_steps = 5e5

        if ask_action==2:
            # print ('stop asking')
            self.agent_asked_for_help = False
            self.max_expert_span = max(self.expert_action_span,self.max_expert_span)
            self.expert_action_span = 0 ##reset counter

        if ask_action==0:
            # print ('do nothing')
            self.asked_init_help_flag = True

        if ask_action==3:
            # print ('ask policy called END')
            # self._took_end_action = True
            # self._success = self._is_goal_in_range()
            # if not self._success:
            #     self.false_stop = 1
            # self.last_action_success = self._success
            self.agent_asked_for_help = False
            action_str = END
        '''   
                

        action = action['nav_action']
        assert isinstance(action, int)
        action = cast(int, action)

        if self.agent_asked_for_help:
            self.num_steps_expert+=1

        action_str = self.class_action_names()[action]

        if self.mirror:
            if action_str == ROTATE_RIGHT:
                action_str = ROTATE_LEFT
            elif action_str == ROTATE_LEFT:
                action_str = ROTATE_RIGHT

        self.task_info["taken_actions"].append(action_str)
        self.task_info["taken_ask_actions"].append(ask_action_str)


        if action_str == END:

            if self.expert_took_step:
                self.expert_ends_traj = True 
            # if ask_action==3:
            #     print ('logic error in ask action END')
            #     exit()
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            if not self._success:
                self.false_stop = 1
            self.last_action_success = self._success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)

        if ask_action==0:
            self.expert_took_step = True  
        else:
            self.expert_took_step = False     


        if len(self.path) > 1:
            self.travelled_distance += IThorEnvironment.position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            frame = self.env.current_frame.copy()
        elif mode == "depth":
            frame = self.env.current_depth.copy()
        else:
            raise NotImplementedError(f"Mode '{mode}' is not supported.")

        if self.mirror:
            frame = frame[:, ::-1, :].copy()  # horizontal flip
            # print("mirrored render")
        return frame

    def _is_goal_in_range(self) -> bool:
        return any(
            o["objectType"] == self.task_info["object_type"]
            for o in self.env.visible_objects()
        )

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.env.distance_to_object_type(
            self.task_info["object_type"]
        )

        # Ensuring the reward magnitude is not greater than the total distance moved
        max_reward_mag = 0.0
        if len(self.path) >= 2:
            p0, p1 = self.path[-2:]
            max_reward_mag = math.sqrt(
                (p0["x"] - p1["x"]) ** 2 + (p0["z"] - p1["z"]) ** 2
            )

        if self.reward_configs.get("positive_only_reward", False):
            if geodesic_distance > 0.5:
                rew = max(self.closest_geo_distance - geodesic_distance, 0)
        else:
            if (
                self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5
            ):  # (robothor limits)
                rew += self.last_geodesic_distance - geodesic_distance

        self.last_geodesic_distance = geodesic_distance
        self.closest_geo_distance = min(self.closest_geo_distance, geodesic_distance)

        return (
            max(min(rew, max_reward_mag), -max_reward_mag,)
            * self.reward_configs["shaping_weight"]
        )

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()

        '''
        if self.help_asked_at_all and (self.asked_init_help_flag is False):
            # print ('give initial ask penalty')
            if not self.penalty_given_once:
                # print ('given initial ask')
                reward += self.reward_configs['penalty_for_init_ask']
                self.penalty_given_once = True 
            else:
                # print ('given recurring')
                reward += self.reward_configs['penalty_for_ask_recurring']    
            self.asked_init_help_flag = True
        '''
         ## for 2 actions
        if self.help_asked_at_all:
            if not self.penalty_given_once:
                reward += self.reward_configs['penalty_for_init_ask']
                self.penalty_given_once = True     
           
        if self.agent_asked_for_help:
            # print ('step ask penalty')
            reward += self.reward_configs['penalty_for_step_ask']

        if self._took_end_action:
            if self._success:
                reward += self.reward_configs["goal_success_reward"]
            else:
                reward += self.reward_configs["failed_stop_reward"]
                
        elif self.num_steps_taken() + 1 >= self.max_steps:
            self.false_stop=1
            reward += self.reward_configs['failed_stop_reward']
            # reward += self.reward_configs.get("reached_max_steps_reward", 0.0)

        self._rewards.append(float(reward))
        return float(reward)

    def get_observations(self, **kwargs) -> Any:
        obs = self.sensor_suite.get_observations(env=self.env, task=self)
        if self.mirror:
            for o in obs:
                if ("rgb" in o or "depth" in o) and isinstance(obs[o], np.ndarray):
                    if (
                        len(obs[o].shape) == 3
                    ):  # heuristic to determine this is a visual sensor
                        obs[o] = obs[o][:, ::-1, :].copy()  # horizontal flip
                    elif len(obs[o].shape) == 2:  # perhaps only two axes for depth?
                        obs[o] = obs[o][:, ::-1].copy()  # horizontal flip
        return obs

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super(ObjectNavTask, self).metrics()
        if self._all_metadata_available:
            dist2tget = self.env.distance_to_object_type(self.task_info["object_type"])

            spl = spl_metric(
                success=self._success,
                optimal_distance=self.optimal_distance,
                travelled_distance=self.travelled_distance,
            )

            expert_action_ratio = self.num_steps_expert/self.num_steps_taken()

            metrics = {
                **metrics,
                "success": self._success,
                "total_reward": np.sum(self._rewards),
                "dist_to_target": dist2tget,
                "part_taken_over_by_expert":expert_action_ratio,
                "false_done_actions":self.false_stop,
                "helped_asked_at_all":self.help_asked_at_all,
                "longest_span_of_expert":self.max_expert_span,
                "expert_ends_traj":self.expert_ends_traj,
                "spl": 0 if spl is None else spl,
            }
        return metrics

    def query_expert(self, end_action_only: bool = False, **kwargs) -> Tuple[int, bool]:

        if not self.agent_asked_for_help:
            return 0,False

        '''
        noise_control = np.random.choice([0,1],p=[0.8,0.2])
        if noise_control==0:
            action_idx = np.random.choice([0,1,2,4,5],p=[1/5]*5)
            #return self.class_action_names().index(action_idx), True
            return action_idx, True    
        '''    

        if self._is_goal_in_range():
            return self.class_action_names().index(END), True

        if end_action_only:
            return 0, False
        else:
            try:
                self.env.step(
                    {
                        "action": "ObjectNavExpertAction",
                        "objectType": self.task_info["object_type"],
                    }
                )
            except ValueError:
                raise RuntimeError(
                    "Attempting to use the action `ObjectNavExpertAction` which is not supported by your version of"
                    " AI2-THOR. The action `ObjectNavExpertAction` is experimental. In order"
                    " to enable this action, please install the (in development) version of AI2-THOR. Through pip"
                    " this can be done with the command"
                    " `pip install -e git+https://github.com/allenai/ai2thor.git@7d914cec13aae62298f5a6a816adb8ac6946c61f#egg=ai2thor`."
                )
            if self.env.last_action_success:
                expert_action: Optional[str] = self.env.last_event.metadata[
                    "actionReturn"
                ]
                if isinstance(expert_action, str):
                    if self.mirror:
                        if expert_action == "RotateLeft":
                            expert_action = "RotateRight"
                        elif expert_action == "RotateRight":
                            expert_action = "RotateLeft"

                    return self.class_action_names().index(expert_action), True
                else:
                    # This should have been caught by self._is_goal_in_range()...
                    return 0, False
            else:
                return 0, False


class NavToPartnerTask(Task[RoboThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT)

    def __init__(
        self,
        env: RoboThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_configs: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs

        assert self.env.agent_count == 2, "NavToPartnerTask only defined for 2 agents!"

        pose1 = self.env.agent_state(0)
        pose2 = self.env.agent_state(1)
        self.last_geodesic_distance = self.env.distance_cache.find_distance(
            self.env.scene_name,
            {k: pose1[k] for k in ["x", "y", "z"]},
            {k: pose2[k] for k in ["x", "y", "z"]},
            self.env.distance_from_point_to_point,
        )

        self.task_info["followed_path1"] = [pose1]
        self.task_info["followed_path2"] = [pose2]
        self.task_info["action_names"] = self.class_action_names()

    @property
    def action_space(self):
        return gym.spaces.Tuple(
            [
                gym.spaces.Discrete(len(self._actions)),
                gym.spaces.Discrete(len(self._actions)),
            ]
        )

    def reached_terminal_state(self) -> bool:
        return (
            self.last_geodesic_distance <= self.reward_configs["max_success_distance"]
        )

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Tuple[int, int]) -> RLStepResult:
        assert isinstance(action, tuple)
        action_str1 = self.class_action_names()[action[0]]
        action_str2 = self.class_action_names()[action[1]]

        self.env.step({"action": action_str1, "agentId": 0})
        self.last_action_success1 = self.env.last_action_success
        self.env.step({"action": action_str2, "agentId": 1})
        self.last_action_success2 = self.env.last_action_success

        pose1 = self.env.agent_state(0)
        self.task_info["followed_path1"].append(pose1)
        pose2 = self.env.agent_state(1)
        self.task_info["followed_path2"].append(pose2)

        self.last_geodesic_distance = self.env.distance_cache.find_distance(
            self.env.scene_name,
            {k: pose1[k] for k in ["x", "y", "z"]},
            {k: pose2[k] for k in ["x", "y", "z"]},
            self.env.distance_from_point_to_point,
        )

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={
                "last_action_success": [
                    self.last_action_success1,
                    self.last_action_success2,
                ],
                "action": action,
            },
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return tile_images(self.env.current_frames)
        elif mode == "depth":
            return tile_images(self.env.current_depths)

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        if self.reached_terminal_state():
            reward += self.reward_configs["success_reward"]

        return reward  # reward shared by both agents (no shaping)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        return {
            **super().metrics(),
            "success": self.reached_terminal_state(),
        }


