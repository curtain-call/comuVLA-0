import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import base64
import json
from pathlib import Path
from typing import Iterator, Tuple, Optional
import numpy as np
from PIL import Image
from io import BytesIO
import openai


# ====== 配置部分 ======
OPENAI_API_KEY = "sk-vqjPBf0ZmPZltUMYNCVC16YsbOoGKXKerIF6QuTy0VONFTEz"
OPENAI_BASE_URL = "https://chatapi.mssctl.com/v1"

client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)
# =====================

def _iter_lerobot_frames(repo_id: str, root: str) -> Iterator[dict]:
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    ds = LeRobotDataset(repo_id, root=root)
    # 逐帧遍历 train split
    for i in range(len(ds)):
        rec = ds[i]
        # rec keys 兼容 v2 schema
        yield {
            "index": int(rec["index"]),
            "episode_index": int(rec["episode_index"]),
            "frame_index": int(rec["frame_index"]),
            "task_index": int(rec["task_index"]),
            "image": np.asarray(rec["observation.images.image"]),  # [256,256,3], uint8
        }, meta.tasks  # tasks: {task_index: str}

def _build_first_spec_prompt(tasks: dict, task_index: int) -> str:
    t = str(tasks.get(task_index, "")).strip()
    # 修改问题
    t = f"You are given the FIRST frame and the task '{t}'. Describe the desired FINAL successful outcome state as seen in the image, not the current state in one sentence."
    prompt = f"""Task: {t}

You are viewing the INITIAL state of a robotic manipulation task. Your goal is to describe what the FINAL SUCCESSFUL completion would look like visually.

Requirements:
1. Focus on OBSERVABLE changes: object positions, orientations, spatial relationships
2. Use SPECIFIC spatial terms: "on top of", "inside", "aligned with", "touching", "separated by"
3. Mention KEY OBJECTS and their final states
4. Be CONCISE but PRECISE (1-2 sentences max)
5. Avoid mentioning the robot/gripper actions - only describe the end result

Example format: "The [object] should be placed [spatial relationship] the [target location], with [specific orientation/configuration details]."

Question: What should the final successful state look like visually?
Answer:"""
    if not t:
        t = "What is on the table?"
    return prompt

def _build_comprehensive_feasibility_prompt(tasks: dict, task_index: int, success_spec: str) -> str:
    """构造综合可行性评估的 prompt，考虑多个关键因素"""
    task_name = str(tasks.get(task_index, "")).strip()
    
    prompt = f"""Task: {task_name}
Expected Success State: {success_spec}

You are an expert in robotic manipulation analyzing the current scene. Evaluate the FEASIBILITY of completing this task from the current state.

Please analyze these CRITICAL FACTORS systematically:

1. **GRIPPER-OBJECT INTERACTION**:
   - Grip quality: Is the gripper properly grasping the target object? (secure/loose/failed/no-contact)
   - Contact geometry: Is the contact angle and surface appropriate?
   - Object deformation: Is the object damaged or deformed due to gripping?

2. **OBJECT STATE & INTEGRITY**:
   - Object condition: Is the target object intact and manipulable?
   - Object orientation: Is the object oriented correctly for the next steps?
   - Object stability: Is the object in a stable configuration?

3. **SPATIAL & GEOMETRIC CONSTRAINTS**:
   - Reachability: Are target locations within the robot's workspace?
   - Path clearance: Are there obstacles blocking the required motion path?
   - Collision risks: Could the current configuration lead to collisions?

4. **TASK LOGIC & SEQUENCING**:
   - Prerequisites: Are necessary preconditions satisfied?
   - Operation sequence: Is the robot following the correct order of operations?
   - Critical dependencies: Are key relationships between objects maintained?

5. **ENVIRONMENTAL FACTORS**:
   - Interference objects: Do other objects block or complicate the task?
   - Surface conditions: Are contact surfaces suitable (friction, stability)?
   - Visibility: Are critical areas visible and accessible?

Based on your analysis, provide:
1. **Current Status**: Brief description of what you observe
2. **Critical Issues**: List any factors that could prevent success (or "None" if no issues)
3. **Feasibility Assessment**: Can this task still be completed? (Highly Likely/Possible/Unlikely/Impossible)
4. **Key Requirements**: What must happen next for success?

Format your response as:
Status: [brief current state description]
Issues: [list critical problems or "None"]
Feasibility: [Highly Likely/Possible/Unlikely/Impossible]
Next Steps: [what needs to happen next]

Question: Analyze the current feasibility of completing this robotic manipulation task.
Answer:"""
    
    return prompt

def _gemini_qa(image_np: np.ndarray, prompt: str) -> Tuple[str, Optional[float]]:
    # 复用已有 gpt4o_caption 发送逻辑，但输出 Q/A + JSON。
    # 要求模型严格返回：<自然语言答案>\n{"success_prob": 0..1}
    img = Image.fromarray(image_np)
    img = img.convert("RGB")
    # 轻度复用现有函数：改提示与解析
    sys = "You are an expert in robotic manipulation and visual scene understanding. Provide precise, objective descriptions focusing on spatial relationships and object states. Be concise but thorough in your analysis."
    user_prompt = f"{prompt}\nReturn format:\n<answer line>\n{{\"success_prob\": 0.xx}}"
    desc = gpt4o_caption_with_system(img, sys, user_prompt)  # 下面补充

    # 解析两行输出
    answer = ""
    p = None
    try:
        lines = [l for l in desc.splitlines() if l.strip()]
        answer = lines[0].strip()
        for l in lines[1:]:
            l = l.strip()
            if l.startswith("{") and l.endswith("}"):
                p = json.loads(l).get("success_prob", None)
                break
    except Exception:
        pass
    return answer, p

def gpt4o_caption_with_system(pil_image: Image.Image, system_text: str, user_text: str, max_retries: int, retry_delay: int) -> str:
    # 与 gpt4o_caption 相同底层 client；仅把 system 文案 & user 文案替换
    from io import BytesIO
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="Qwen2.5-VL-72B-Instruct",
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]}
                ],
                max_tokens=300,
                temperature=0.7,  # 降低温度以获得更一致的输出
                top_p=0.9,
                stream=False
            )
            
            result = response.choices[0].message.content.strip()
            
            # 检查返回结果是否为空或过短
            if not result or len(result.strip()) < 10:
                raise ValueError(f"API 返回空或过短内容: '{result}'")
                
            return result
            
        except Exception as e:
            print(f"API 调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # 指数退避
            else:
                print("所有重试均失败，返回错误标记")
                return f"[API_ERROR] 调用失败: {str(e)}"

def annotate_libero_spatial_with_gemini(repo_id: str, root: str, out_jsonl: str,
                                       action_horizon: int = 10, include_first: bool = True):
    """
    每隔 action_horizon 帧做一次标注；可选首帧。
    输出 JSONL：{"index":int,"episode_index":int,"frame_index":int,"task_index":int,"answer":str,"success_prob":float}
    """
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if Path(out_jsonl).exists():
        # 断点续跑
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    seen.add((j["episode_index"], j["frame_index"]))
                except Exception:
                    pass
    success_spec_by_ep = {}
    success_out = Path(out_jsonl).with_name(Path(out_jsonl).stem + "_success_spec.jsonl")
    with open(out_jsonl, "a", encoding="utf-8") as fout, open(success_out, "a", encoding="utf-8") as fspec:

        for rec, tasks in _iter_lerobot_frames(repo_id, root):
            epi, frm = rec["episode_index"], rec["frame_index"]

            # 1) 首帧：生成并缓存 success-spec（只做一次）
            if frm == 0 and epi not in success_spec_by_ep:
                spec_prompt = _build_first_spec_prompt(tasks, rec["task_index"])
                if rec["image"].dtype == np.float32:
                    # 假设 float32 范围是 [0,1]，转换为 [0,255] uint8
                    rec["image"] = (rec["image"] * 255).astype(np.uint8)
                elif rec["image"].dtype != np.uint8:
                    # 其他格式也转为 uint8
                    rec["image"] = rec["image"].astype(np.uint8)
                spec_text = gpt4o_caption_with_system(
                    Image.fromarray(np.transpose(rec["image"],(1,2,0))).convert("RGB"),
                    "You are an expert in robotic manipulation and visual scene understanding. Provide precise, objective descriptions focusing on spatial relationships and object states. Be concise but thorough in your analysis.",
                    spec_prompt,
                    5,
                    2
                )
                success_spec_by_ep[epi] = spec_text
                fspec.write(json.dumps({
                    "episode_index": epi,
                    "task_index": rec["task_index"],
                    "success_spec": spec_text,
                    "prompt": spec_prompt
                }, ensure_ascii=False) + "\n")
                fspec.flush()
                print(f"[success-spec] ep={epi} -> {spec_text}")

            # 2) 帧级问答：仍按 horizon 采样（例如每 H 帧一次）
            if include_first and frm == 0:
                pass
            elif (frm + 1) % action_horizon != 0:
                continue
            
            qa_prompt = _build_comprehensive_feasibility_prompt(tasks, rec["task_index"], success_spec_by_ep[epi])  # 你现有的 VQA 问题构造
            answer, prob = _gemini_qa(rec["image"], qa_prompt)
            payload = {
                "index": rec["index"],
                "episode_index": epi,
                "frame_index": frm,
                "task_index": rec["task_index"],
                "": answer,
                "prompt": qa_prompt,
                # 回填该 episode 的成功状态描述
                "success_spec": success_spec_by_ep.get(epi)
            }
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[annotated] ep={epi} frame={frm} ans={answer} p={prob}")



if __name__ == '__main__':
    annotate_libero_spatial_with_gemini("physical-intelligence/libero_spatial_image", root="/home/zhiyu/mzh/datasets/libero_spatial_image/", out_jsonl="/home/zhiyu/mzh/datasets/libero_spatial_image/first_frame_description.jsonl")
