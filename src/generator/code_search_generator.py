import copy
import json
import asyncio
from pyexpat.errors import messages
from socket import timeout
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid
from omegaconf import DictConfig
import traceback
import ray
import requests
from pathlib import Path
import os
import ast
import time
from datetime import datetime
import numpy as np
from collections import defaultdict

import re
import signal
from contextlib import contextmanager

import gcsfs
import fsspec

from skyrl_train.generators.skyrl_gym_generator import (
    SkyRLGymGenerator,
    GeneratorOutput,
    GeneratorInput,
)
from skyrl_train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.generators.utils import (
    get_rollout_metrics,
    encode_messages_subset,
)
from openhands.tools.preset.default import get_default_agent

from openhands.sdk.conversation.response_utils import get_agent_final_response
from openhands.workspace import DockerWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.tools.preset.planning import get_planning_tools
from openhands.tools.terminal import TerminalTool
from openhands.sdk.tool import Tool, register_tool

# Import built-in tools to auto-register them
import openhands.tools.glob
import openhands.tools.grep
import openhands.tools.apply_patch
import openhands.tools.file_editor
import openhands.tools.planning_file_editor
import openhands.tools.task_tracker
import openhands.tools.browser_use
import openhands.tools.delegate
import openhands.tools.preset
import openhands.tools.tom_consult
from openhands.sdk import (
    Agent,
    LLM,
    Event,
    Conversation,
    RemoteConversation,
    LLMConvertibleEvent,
    get_logger,
)

from src.prompts.prompt_builder import get_instruction
from src.utils.instance import clone_instance
from src.agent.agent import CustomAgent

from src.rewards import get_reward_function
<<<<<<< Updated upstream
from src.tools import TOOL_REGISTRY, DEFAULT_OPENHANDS_TOOLS, import_openhands_tool
=======
from src.tools import TOOL_REGISTRY, DEFAULT_OPENHANDS_TOOLS
>>>>>>> Stashed changes

from src.metrics.efficiency_metrics import compute_all_efficiency_metrics
from src.metrics.trajectory_metrics import compute_trajectory_metrics

import logging
import signal

logger = get_logger(__name__)
# logger.setLevel(logging.WARNING)
logger.setLevel(logging.ERROR)

file_path = os.path.dirname(__file__)

@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    litellm_base_url: dict,
    generator_cfg: DictConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: Union[TrajectoryID, Any],
    global_step: int,
    training_phase: Union[TrainingPhase, Any],
):

    instance_id = instance["instance_id"]
    repo_name = instance["repo"]
    commit_id = instance["base_commit"]
    
    # Avoid collisions in /tmp testbed directories
    uuid_str = str(uuid.uuid4())[:8]
    workspace = Path(os.environ.get("TESTBED_ROOT", "/tmp/testbed")) / uuid_str
    status, working_dir = clone_instance(repo_name, commit_id, instance_id, workspace)

    if training_phase == "eval":
        temperature = 0.6
    else:
        temperature = 1.0

    final_message = ""
    messages = []

    # Ray worker processes may start with an empty OpenHands tool registry.
    # Import built-in tools here to ensure self-registration in *this* process.
    from openhands.sdk.tool import registry as _tool_registry  # noqa: F401
    import openhands.tools.glob  # noqa: F401
    import openhands.tools.grep  # noqa: F401
    import openhands.tools.terminal  # noqa: F401

    # Register custom tools (built-in tools don't need registration)
    for tool_name in generator_cfg.tools:
<<<<<<< Updated upstream
        # Import OpenHands tools to trigger their registration
        if tool_name in DEFAULT_OPENHANDS_TOOLS:
            import_openhands_tool(tool_name)
        # Register custom tools from our registry
        elif tool_name in TOOL_REGISTRY:
            register_tool(tool_name, TOOL_REGISTRY[tool_name])
        else:
            raise ValueError(f"Tool {tool_name} does not exist in the registry or default OpenHands tools")
=======
        if tool_name in TOOL_REGISTRY:
            # Custom tool - needs registration
            register_tool(tool_name, TOOL_REGISTRY[tool_name])
        elif tool_name not in DEFAULT_OPENHANDS_TOOLS:
            # Not a built-in tool and not in custom registry
            raise ValueError(f"Tool {tool_name} does not exist in the registry")
>>>>>>> Stashed changes

    tools = [
        Tool(name=tool_name) for tool_name in generator_cfg.tools
    ]

    # Get prompt paths from config (path-independent)
    prompts_base_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")
    system_prompt_path = os.path.join(prompts_base_dir, generator_cfg.prompts.system_prompt)
    user_prompt_path = os.path.join(prompts_base_dir, generator_cfg.prompts.user_prompt)

    agent = CustomAgent(
        llm=LLM(
            usage_id="agent",
            model=litellm_model_name,
            base_url=litellm_base_url,
            api_key="sk-xxx",
            temperature=temperature,
            litellm_extra_body={
                "return_token_ids": True,
                "include_stop_str_in_output": True,
            }
        ),
        tools=tools,
        security_analyzer=None,
        system_prompt_filename=system_prompt_path
    )

    # Create conversation with max_iteration_per_run limit
    conversation = None
    try:
        conversation = Conversation(
            agent=agent,
            max_iteration_per_run=10,
            visualizer=None,
            workspace=str(working_dir),
        )
        input_message = get_instruction(instance, user_prompt_path, str(working_dir))
        conversation.send_message(input_message)
    except Exception:
        # Conversation initialization can fail early (e.g., tool registry mismatch).
        # Ensure we don't leak /tmp testbed clones in that case.
        try:
            if workspace.exists():
                os.system(f"rm -rf {str(workspace)}")
        except Exception:
            logger.error(f"Error removing workspace {str(workspace)}:\n{traceback.format_exc()}")
        try:
            if conversation is not None:
                conversation.close()
        except Exception:
            logger.error(f"Error closing conversation:\n{traceback.format_exc()}")
        raise

    logger.info("Conversation Starting")

    # Capture start time
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    try:
        conversation.run()

        messages = list(map(lambda event: event.model_dump(), conversation.state.events))
        final_message = get_agent_final_response(conversation.state.events)
    finally:
        # Always clean up the cloned workspace, even if Conversation init/run fails.
        try:
            if workspace.exists():
                os.system(f"rm -rf {str(workspace)}")
        except Exception:
            # Avoid passing exception objects through loguru multiprocessing handler
            logger.error(
                f"Error removing workspace {str(workspace)}:\n{traceback.format_exc()}"
            )
        try:
            conversation.close()
        except Exception:
            logger.error(f"Error closing conversation:\n{traceback.format_exc()}")

    logger.info("Conversation Finished")

    # Capture end time
    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    wall_clock_duration = end_time - start_time

    additional_attr = {
        "wall_clock_duration": wall_clock_duration,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp
    }

    return messages, final_message, additional_attr


class CodeSearchGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        # Call parent constructor first
        super().__init__(
            generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name
        )

        self.http_endpoint_host = generator_cfg.get(
            "http_endpoint_host", "127.0.0.1"
        )
        self.http_endpoint_port = generator_cfg.get(
            "http_endpoint_port", 8000
        )
        self.base_url = f"http://{self.http_endpoint_host}:{self.http_endpoint_port}/v1/"
        logger.info(f"Using CodeSearchGenerator with model {model_name} at {self.base_url}")
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        # self.litellm_model_name = "openai/" + self.model_name
        self.litellm_model_name = "litellm_proxy/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError(
                "OpenhandsGenerator doesn't support custom chat template"
            )

    async def code_search_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]], Optional[Dict[str, Any]]]:
        # sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        instance = env_extras
        error = None
        try:
            messages, final_message, additional_attr = await init_and_run.remote(
                instance,
                self.litellm_model_name,
                # sweagent_config,
                self.base_url,
                self.generator_cfg,
                # env_extras["data_source"],
                "swe-gym",
                sampling_params,
                trajectory_id,
                batch_metadata.global_step,
                batch_metadata.training_phase,
            )
        except Exception as e:
            # Avoid passing exception objects through loguru multiprocessing handler
            logger.error(f"Error in starting conversation: {e}\n{traceback.format_exc()}")
            # TODO properly handle this
            error = str(e) + "\n" + traceback.format_exc()
            messages = []
            final_message = ""
            additional_attr = {
                "wall_clock_duration": 0.0,
                "start_timestamp": None,
                "end_timestamp": None
            }

        # print("=" * 100)
        # print("Conversation finished. Got the following LLM messages:")
        # for i, message in enumerate(messages):
        #     print(f"Message {i}: {str(message)[:100]}")
        # print("Final message:", final_message)

        # Reward Manager
        reward = 0
        reward_dict = {}

        for reward_fn_args in self.generator_cfg.reward:
            try:
                input_args = {
                    "final_message": final_message,
                    "messages": messages,
                    "instance": instance,
                }

                reward_fn = get_reward_function(reward_fn_args["fn"])

                input_args = {
                    **input_args, 
                    **reward_fn_args.get("args", {})
                    }

                reward_outputs = reward_fn(**input_args)
                if isinstance(reward_outputs, tuple):
                    reward_value, reward_items = reward_outputs
                else:
                    reward_value = reward_outputs
                    reward_items = {reward_fn_args["fn"]: reward_value}
            except Exception as e:
                logger.error(f"Error in computing reward {reward_fn_args['fn']}: {e}", exc_info=True)
                reward_value = 0.0
                reward_items = {reward_fn_args["fn"]: reward_value}

            reward += reward_value

            reward_dict = {
                **reward_dict,
                **reward_items,
            }

        if final_message == "":
            reward = -10.0

        logger.info(f"Reward details: {reward_dict}, Total reward: {reward}")

        # Compute Trajectory Metrics
        efficiency_metrics = compute_all_efficiency_metrics(
            messages=messages,
            **additional_attr,
        )

        trajectory_metrics = compute_trajectory_metrics(messages)

        metrics_dict = {
            **efficiency_metrics,
            **trajectory_metrics
        }

        print(f"Trajectory metrics: {metrics_dict}")

        token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
        rollout_list = []
        if len(token_messages) > 0:
            for idx, message in enumerate(token_messages):
                current_prompt_ids = message["prompt_token_ids"]
                current_response_ids = message["response_token_ids"]
                step_reward = reward

                rollout_list.append(
                    (
                        current_response_ids,
                        step_reward,
                        "complete",
                        [1]*len(current_response_ids),
                        current_prompt_ids,
                        None,
                        trajectory_metrics
                    )
                )

        else:
            response_ids = [151643]
            stop_reason = "error"
            loss_mask = [1]
            initial_input_ids = [151643]
            trajectory_metrics = {}  # Empty metrics for error case
            rollout_list.append(
                (response_ids, reward, stop_reason, loss_mask, initial_input_ids, None, trajectory_metrics)
            )

        # Add "/" at the end of traj_dir if not present
        if not self.generator_cfg.traj_dir.endswith("/"):
            self.generator_cfg.traj_dir += "/"

        path = self.generator_cfg.traj_dir + f"step_{batch_metadata.global_step}/{batch_metadata.training_phase}/"
        # Check if traj_dir is a gcs path
        if path.startswith("gs://"):
            use_gcs = True
            fs = gcsfs.GCSFileSystem()
        else:
            use_gcs = False
            fs = fsspec.filesystem("file")
            # Pre-create directory to avoid race conditions with parallel workers
            os.makedirs(path, exist_ok=True)
        
        instance_id = env_extras["instance_id"]

        if error is not None:
            filename = f"{instance_id}_{trajectory_id.repetition_id}.error"
            filename_path = path + filename
            print(f"Saving error to {filename_path}")
            if use_gcs == False:
                os.makedirs(os.path.dirname(filename_path), exist_ok=True)
            with fs.open(filename_path, "w", auto_mkdir=True) as f:
                f.write(error)
        else:
            filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
            filename_path = path + filename

            if use_gcs == False:
                os.makedirs(os.path.dirname(filename_path), exist_ok=True)

            # get everything between ```` with regex
            raw_final_message = final_message
            matches = re.findall(r"```(.*?)```", final_message, re.DOTALL)
            parsed_final_message = matches[0] if matches else final_message

            result_dict = {
                "instance_id": instance_id,
                "target": env_extras["target"],
                "total_reward": reward,
                "reward_dict": reward_dict,
                "parsed_final_message": parsed_final_message,
                "raw_final_message": raw_final_message,
                "messages": messages,
                "metrics_dict": metrics_dict,
            }

            print(f"Saving trajectory to {filename_path}")
            with fs.open(filename_path, "w", auto_mkdir=True) as f:
                json.dump(result_dict, f, indent=2) #, sort_keys=True, ensure_ascii=False)

        return [rollout_list, reward_dict, metrics_dict]

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.backend, self.generator_cfg.sampling_params
        )

        task_rollouts = []
        for i in range(len(prompts)):
            rollout = self.code_search_loop(
                    prompts[i],
                    env_extras[i],
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i],
                    batch_metadata=batch_metadata,
                )
            
            task_rollouts.append(rollout)

        # Use return_exceptions=True to handle partial failures gracefully
        collected_task_rollouts = await asyncio.gather(*task_rollouts, return_exceptions=True)

        # Separate successful results from exceptions
        all_outputs = []
        rewards_dict = []
        metrics_dict = []
        failed_count = 0
        
        for i, rollout in enumerate(collected_task_rollouts):
            if isinstance(rollout, Exception):
                # Log the error but continue with other rollouts
                logger.error(
                    f"Failed to generate rollout for prompt {i} "
                    f"(trajectory_id={trajectory_ids[i]}): {rollout}",
                    exc_info=rollout
                )
                failed_count += 1
                # Create a minimal error rollout to maintain batch structure
                error_response_ids = [151643]  # Default error token
                error_rollout = [
                    (error_response_ids, -10.0, "error", [1], [151643], None, {})
                ]
                all_outputs.append(error_rollout)
                rewards_dict.append({})
                metrics_dict.append({})
            else:
                all_outputs.append(rollout[0])
                rewards_dict.append(rollout[1])
                metrics_dict.append(rollout[2])
        
        if failed_count > 0:
            logger.warning(
                f"Generation completed with {failed_count}/{len(prompts)} failures. "
                f"Success rate: {(len(prompts) - failed_count) / len(prompts) * 100:.1f}%"
            )

        # Only take the last step output for each trajectory to match prompt count
        responses = [step_outputs[-1][0] for step_outputs in all_outputs]
        rewards = [step_outputs[-1][1] for step_outputs in all_outputs]
        stop_reasons = [step_outputs[-1][2] for step_outputs in all_outputs]
        loss_masks = [step_outputs[-1][3] for step_outputs in all_outputs]
        prompt_token_ids = [step_outputs[-1][4] for step_outputs in all_outputs]

        # Since we only return the last step for each trajectory
        out_trajectory_ids = []
        is_last_step = []
        for i in range(len(all_outputs)):
            step_outputs = all_outputs[i]
            out_trajectory_id = copy.deepcopy(trajectory_ids[i])
            out_trajectory_id.step = len(step_outputs) - 1  # Set to the last step index
            out_trajectory_ids.append(out_trajectory_id.instance_id)
            is_last_step.append(True)  # Always True since we only return last steps

        if not len(responses):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards)

        tracked_metrics = {}

        # Aggregate Rewards and Metrics
        for tracker_name, tracker_dict in zip(
            ["reward", "metrics"], [rewards_dict, metrics_dict]
        ):
            for tracker_dict_item in tracker_dict:
                for k, v in tracker_dict_item.items():
                    # Check if v is numeric
                    if not isinstance(v, (int, float)):
                        continue
                    if f"{tracker_name}/{k}" not in tracked_metrics:
                        tracked_metrics[f"{tracker_name}/{k}"] = []
                    tracked_metrics[f"{tracker_name}/{k}"].append(v)
        
        # Average all tracked metrics
        for k, v in tracked_metrics.items():
            tracked_metrics[k] = sum(v) / len(v)

        # IMPORTANT: skyrl_train only logs generator_output["rollout_metrics"] to tracker/wandb.
        # Merge our custom aggregated metrics into rollout_metrics so they are visible in wandb.
        rollout_metrics = {
            **rollout_metrics,
            **tracked_metrics,
        }

        generator_output: GeneratorOutput = {
            "trajectory_ids": out_trajectory_ids,
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
            "is_last_step": is_last_step,
        }

        return generator_output