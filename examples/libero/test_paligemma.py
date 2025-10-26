# examples/libero/test_paligemma_hf.py
import dataclasses
import logging
import pathlib
import numpy as np
import tyro
import pickle
from PIL import Image
import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import time

@dataclasses.dataclass
class HFInferArgs:
    """HuggingFace 推理配置"""
    data_file: str = "libero_observation_data/libero_observations.pkl"
    output_dir: str = "paligemma_hf_results"
    
    # 模型配置
    model_name: str = "/home/zhiyu/mzh/openpi/checkpoints/paligemma_hf"
    use_gpu: bool = True
    max_new_tokens: int = 50
    
    # 生成配置
    do_sample: bool = False  # 贪心解码
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

class HFPaliGemmaInferencer:
    """HuggingFace PaliGemma 推理器"""
    
    def __init__(self, model_name: str, use_gpu: bool = True):
        print(f"=== 初始化 HuggingFace PaliGemma ===")
        print(f"模型: {model_name}")
        
        # 设备配置
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        if torch.cuda.is_available() and use_gpu:
            print(f"CUDA 设备数量: {torch.cuda.device_count()}")
            print(f"当前 CUDA 设备: {torch.cuda.current_device()}")
            print(f"设备名称: {torch.cuda.get_device_name()}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # 加载模型和处理器
        print("加载模型和处理器...")
        try:
            self.processor = PaliGemmaProcessor.from_pretrained(model_name)
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if use_gpu else torch.float32,
                device_map="auto" if use_gpu else None
            )
            
            if not use_gpu:
                self.model = self.model.to(self.device)
            
            print("✅ 模型加载完成")
            print(f"   - 处理器词汇表大小: {len(self.processor.tokenizer)}")
            print(f"   - 模型参数量: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def generate_caption(self, image: np.ndarray, prompt: str = "caption en:", **kwargs) -> str:
        """生成图像描述"""
        
        try:
            print(f"\n--- 生成描述 ---")
            print(f"Prompt: '{prompt}'")
            print(f"图像形状: {image.shape}")
            
            # 转换图像格式
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    # 假设输入是 [0, 1] 范围的 float
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            print(f"PIL 图像大小: {image.size}")
            
            # 处理输入
            start_time = time.time()
            inputs = self.processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            process_time = time.time() - start_time
            print(f"输入处理时间: {process_time:.3f}s")
            
            # 打印输入信息
            print(f"输入 tokens 形状: {inputs['input_ids'].shape}")
            print(f"输入 tokens: {inputs['input_ids'][0].tolist()}")
            
            # 解码输入以验证
            input_text = self.processor.decode(inputs['input_ids'][0], skip_special_tokens=False)
            print(f"输入文本: '{input_text}'")
            
            # 生成
            print("开始生成...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_new_tokens', 50),
                    do_sample=kwargs.get('do_sample', False),
                    temperature=None,
                    top_k=None,
                    top_p=None,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    repetition_penalty=1.0,
                    length_penalty=1.0,
                    early_stopping=True,
                )
            
            generation_time = time.time() - start_time
            print(f"生成时间: {generation_time:.3f}s")
            
            # 解码完整输出
            full_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            print(f"完整输出: '{full_text}'")
            
            # 提取生成的部分（移除输入 prompt）
            if prompt and full_text.startswith(prompt):
                generated_text = full_text[len(prompt):].strip()
            else:
                # 尝试更智能的提取
                generated_text = full_text
                for possible_prompt in [prompt, prompt.lower(), prompt.capitalize()]:
                    if possible_prompt and generated_text.startswith(possible_prompt):
                        generated_text = generated_text[len(possible_prompt):].strip()
                        break
            
            print(f"生成的文本: '{generated_text}'")
            
            # 打印生成的 token 信息
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            print(f"生成的 tokens: {generated_tokens.tolist()}")
            print(f"生成 token 数量: {len(generated_tokens)}")
            
            return generated_text if generated_text else "[NO GENERATION]"
            
        except Exception as e:
            print(f"生成过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return f"[ERROR: {str(e)}]"

def test_paligemma_hf(args: HFInferArgs):
    """使用 HuggingFace PaliGemma 测试预存数据"""
    
    # 1) 加载数据
    print(f"=== 加载 Libero 观测数据 ===")
    data_file = pathlib.Path(args.data_file)
    if not data_file.exists():
        print(f"❌ 数据文件不存在: {data_file}")
        print("请先运行 collect_libero_data.py 收集数据")
        return
    
    with open(data_file, "rb") as f:
        collected_data = pickle.load(f)
    
    print(f"✅ 成功加载 {len(collected_data)} 个任务的数据")
    
    # 2) 初始化 PaliGemma
    try:
        model = HFPaliGemmaInferencer(args.model_name, args.use_gpu)
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3) 创建输出目录
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 4) 对每个任务进行推理
    all_results = []
    
    for i, task_data in enumerate(collected_data):
        print(f"\n{'=' * 60}")
        print(f"推理任务 {i + 1}/{len(collected_data)}")
        print(f"{'=' * 60}")
        
        task_id = task_data["task_id"]
        task_description = task_data["task_description"]
        
        print(f"任务 ID: {task_id}")
        print(f"任务描述: {task_description}")
        
        # 获取处理后的图像
        agentview_img = task_data["images"]["agentview_processed"]
        wrist_img = task_data["images"]["wrist_processed"]
        
        print(f"AgentView 图像形状: {agentview_img.shape}")
        print(f"Wrist 图像形状: {wrist_img.shape}")
        
        # 测试不同的 prompt
        test_prompts = [
            # f"How to determine whether task '{task_description}' succeeded or failed? Answer: ",
            f"What is on the table? Answer: ",
        ]
        
        task_results = {
            "task_id": task_id,
            "task_description": task_description,
            "agentview_results": [],
            "wrist_results": []
        }
        
        # 对 AgentView 图像进行推理
        print("\n--- AgentView 图像推理 ---")
        for j, prompt in enumerate(test_prompts):
            print(f"\n测试 prompt {j+1}: '{prompt}'")
            
            try:
                caption = model.generate_caption(
                    agentview_img, 
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                print(f"✅ 结果: '{caption}'")
                task_results["agentview_results"].append((prompt, caption, "SUCCESS"))
            except Exception as e:
                print(f"❌ 错误: {e}")
                task_results["agentview_results"].append((prompt, f"[ERROR: {str(e)}]", "ERROR"))
        
        # 对 Wrist 图像进行推理
        print("\n--- Wrist 图像推理 ---")
        for j, prompt in enumerate(test_prompts):
            print(f"\n测试 prompt {j+1}: '{prompt}'")
            
            try:
                caption = model.generate_caption(
                    wrist_img, 
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                print(f"✅ 结果: '{caption}'")
                task_results["wrist_results"].append((prompt, caption, "SUCCESS"))
            except Exception as e:
                print(f"❌ 错误: {e}")
                task_results["wrist_results"].append((prompt, f"[ERROR: {str(e)}]", "ERROR"))
        
        all_results.append(task_results)
        
        # 保存任务结果
        with open(output_dir / f"task_{task_id}_results.txt", "w", encoding='utf-8') as f:
            f.write(f"Task: {task_description}\n")
            f.write(f"Task ID: {task_id}\n")
            f.write("-" * 60 + "\n")
            
            f.write("\nAgentView 图像结果:\n")
            f.write("-" * 30 + "\n")
            for prompt, result, status in task_results["agentview_results"]:
                f.write(f"Prompt: '{prompt}'\n")
                f.write(f"Status: {status}\n")
                f.write(f"Result: '{result}'\n")
                f.write("-" * 20 + "\n")
            
            f.write("\nWrist 图像结果:\n")
            f.write("-" * 30 + "\n")
            for prompt, result, status in task_results["wrist_results"]:
                f.write(f"Prompt: '{prompt}'\n")
                f.write(f"Status: {status}\n")
                f.write(f"Result: '{result}'\n")
                f.write("-" * 20 + "\n")
        
        # 统计成功率
        agentview_success = sum(1 for _, _, status in task_results["agentview_results"] if status == "SUCCESS")
        wrist_success = sum(1 for _, _, status in task_results["wrist_results"] if status == "SUCCESS")
        
        print(f"\n📊 任务 {task_id} 统计:")
        print(f"   AgentView: {agentview_success}/{len(test_prompts)} 成功")
        print(f"   Wrist: {wrist_success}/{len(test_prompts)} 成功")
    
    # 5) 保存完整结果
    results_file = output_dir / "all_results.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(all_results, f)
    
    # 6) 生成总结报告
    total_agentview_success = sum(
        sum(1 for _, _, status in task["agentview_results"] if status == "SUCCESS")
        for task in all_results
    )
    total_wrist_success = sum(
        sum(1 for _, _, status in task["wrist_results"] if status == "SUCCESS") 
        for task in all_results
    )
    total_tests = len(all_results) * len(test_prompts)
    
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w", encoding='utf-8') as f:
        f.write(f"HuggingFace PaliGemma 推理结果总结\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"模型: {args.model_name}\n")
        f.write(f"设备: {'GPU' if args.use_gpu else 'CPU'}\n")
        f.write(f"测试任务数: {len(all_results)}\n")
        f.write(f"每任务测试 prompt 数: {len(test_prompts)}\n")
        f.write(f"总测试数: {total_tests * 2} (AgentView + Wrist)\n")
        f.write(f"\n生成配置:\n")
        f.write(f"  - max_new_tokens: {args.max_new_tokens}\n")
        f.write(f"  - do_sample: {args.do_sample}\n")
        f.write(f"  - temperature: {args.temperature}\n")
        f.write(f"  - top_k: {args.top_k}\n")
        f.write(f"  - top_p: {args.top_p}\n")
        f.write(f"\n成功率统计:\n")
        f.write(f"AgentView 图像: {total_agentview_success}/{total_tests} ({total_agentview_success/total_tests*100:.1f}%)\n")
        f.write(f"Wrist 图像: {total_wrist_success}/{total_tests} ({total_wrist_success/total_tests*100:.1f}%)\n")
        f.write(f"总体: {total_agentview_success + total_wrist_success}/{total_tests * 2} ({(total_agentview_success + total_wrist_success)/(total_tests * 2)*100:.1f}%)\n")
        
        f.write(f"\n详细任务结果:\n")
        for task in all_results:
            agentview_success = sum(1 for _, _, status in task["agentview_results"] if status == "SUCCESS")
            wrist_success = sum(1 for _, _, status in task["wrist_results"] if status == "SUCCESS")
            f.write(f"任务 {task['task_id']}: AgentView {agentview_success}/{len(test_prompts)}, Wrist {wrist_success}/{len(test_prompts)}\n")
            f.write(f"  描述: {task['task_description']}\n")
    
    print(f"\n✅ 推理完成！")
    print(f"   - 结果保存在: {output_dir}")
    print(f"   - 总结报告: {summary_file}")
    print(f"   - AgentView 成功率: {total_agentview_success}/{total_tests} ({total_agentview_success/total_tests*100:.1f}%)")
    print(f"   - Wrist 成功率: {total_wrist_success}/{total_tests} ({total_wrist_success/total_tests*100:.1f}%)")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tyro.cli(test_paligemma_hf)