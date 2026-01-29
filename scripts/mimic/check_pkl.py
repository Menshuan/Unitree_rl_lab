import pickle
import numpy as np
import argparse

def inspect_pkl_file(file_path):
    """详细查看PKL文件的结构信息"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return None
    
    print(f"📂 文件: {file_path}")
    print("=" * 70)
    
    if isinstance(data, dict):
        print(f"📊 数据结构: 字典 (共 {len(data)} 个键)")
        print("=" * 70)
        
        for i, (key, value) in enumerate(data.items(), 1):
            print(f"\n{i}. 键名: '{key}'")
            print(f"   └─ 类型: {type(value).__name__}")
            
            # 处理不同类型的数据
            if isinstance(value, np.ndarray):
                print(f"   └─ 形状: {value.shape}")
                print(f"   └─ 数据类型: {value.dtype}")
                print(f"   └─ 元素总数: {value.size:,}")
                
                # 显示各维度含义（如果适用）
                if len(value.shape) == 1:
                    print(f"   └─ 描述: 一维数组，长度为 {value.shape[0]}")
                elif len(value.shape) == 2:
                    print(f"   └─ 描述: 二维数组，{value.shape[0]}行 × {value.shape[1]}列")
                    if 'frame' in key.lower() or 'pose' in key.lower():
                        print(f"   └─ 推测: {value.shape[0]}帧，每帧{value.shape[1]}个特征")
                
            elif isinstance(value, list):
                print(f"   └─ 长度: {len(value):,}")
                
                if len(value) > 0:
                    first_item = value[0]
                    print(f"   └─ 元素类型: {type(first_item).__name__}")
                    
                    # 分析列表元素的维度
                    if isinstance(first_item, np.ndarray):
                        print(f"   └─ 元素形状: {first_item.shape}")
                        print(f"   └─ 元素数据类型: {first_item.dtype}")
                    elif isinstance(first_item, list):
                        print(f"   └─ 元素长度: {len(first_item)}")
                        
                        # 如果是多层嵌套列表
                        if len(first_item) > 0 and isinstance(first_item[0], (list, np.ndarray)):
                            second_item = first_item[0]
                            if isinstance(second_item, np.ndarray):
                                print(f"   └─ 二级元素形状: {second_item.shape}")
                            else:
                                print(f"   └─ 二级元素长度: {len(second_item)}")
                    
                    # 检查所有元素是否具有相同结构
                    if len(value) > 1:
                        second_item = value[1]
                        same_type = type(first_item) == type(second_item)
                        if isinstance(first_item, (list, np.ndarray)) and isinstance(second_item, (list, np.ndarray)):
                            same_shape = len(first_item) == len(second_item) if isinstance(first_item, list) else first_item.shape == second_item.shape
                            print(f"   └─ 前两元素类型相同: {same_type}")
                            print(f"   └─ 前两元素维度相同: {same_shape}")
                
            elif isinstance(value, (int, float)):
                print(f"   └─ 值: {value}")
                if 'fps' in key.lower():
                    print(f"   └─ 描述: 帧率 ({value} 帧/秒)")
                elif 'loop' in key.lower():
                    print(f"   └─ 描述: 循环模式 ({value})")
                    
            elif isinstance(value, str):
                print(f"   └─ 值: '{value}'")
                print(f"   └─ 长度: {len(value)} 字符")
                
            elif isinstance(value, bool):
                print(f"   └─ 值: {value}")
                
            else:
                print(f"   └─ 值: {value}")
    
    else:
        # 如果不是字典
        print(f"📊 数据结构: {type(data).__name__}")
        print("=" * 70)
        
        if isinstance(data, np.ndarray):
            print(f"  形状: {data.shape}")
            print(f"  数据类型: {data.dtype}")
            print(f"  元素总数: {data.size:,}")
        elif isinstance(data, list):
            print(f"  长度: {len(data):,}")
            if len(data) > 0:
                print(f"  第一个元素类型: {type(data[0]).__name__}")
    
    return data

def analyze_frames_structure(data):
    """专门分析frames键的结构"""
    if 'frames' not in data:
        print("\n⚠️  没有找到'frames'键")
        return
    
    frames = data['frames']
    print("\n" + "🔍 FRAMES结构详细分析")
    print("=" * 70)
    
    print(f"总帧数: {len(frames):,}")
    
    if 'fps' in data:
        duration = len(frames) / data['fps']
        print(f"帧率(fps): {data['fps']}")
        print(f"动画时长: {duration:.2f}秒")
        if duration > 60:
            print(f"          ({duration/60:.2f}分钟)")
    
    if len(frames) == 0:
        print("⚠️  frames为空")
        return
    
    # 分析第一帧
    first_frame = frames[0]
    print(f"\n第一帧类型: {type(first_frame).__name__}")
    
    if isinstance(first_frame, list):
        print(f"第一帧长度: {len(first_frame)}")
        
        if len(first_frame) > 0:
            first_element = first_frame[0]
            print(f"第一帧元素类型: {type(first_element).__name__}")
            
            if isinstance(first_element, np.ndarray):
                print(f"第一帧元素形状: {first_element.shape}")
                print(f"第一帧元素数据类型: {first_element.dtype}")
            elif isinstance(first_element, list):
                print(f"第一帧元素长度: {len(first_element)}")
            elif isinstance(first_element, (int, float)):
                print(f"第一帧元素示例: {first_element}")
        
        # 检查多帧的一致性
        print(f"\n一致性检查:")
        sample_frames = min(10, len(frames))
        lengths = [len(frame) for frame in frames[:sample_frames] if isinstance(frame, list)]
        
        if lengths:
            unique_lengths = set(lengths)
            if len(unique_lengths) == 1:
                print(f"  前{sample_frames}帧长度一致: {lengths[0]}")
            else:
                print(f"  前{sample_frames}帧长度不一致: {list(unique_lengths)}")
        
        # 显示前几帧的维度信息
        print(f"\n前5帧的维度:")
        for i in range(min(5, len(frames))):
            frame = frames[i]
            if isinstance(frame, list):
                print(f"  帧{i}: 长度={len(frame)}", end="")
                if len(frame) > 0:
                    elem = frame[0]
                    if isinstance(elem, np.ndarray):
                        print(f", 元素形状={elem.shape}")
                    elif isinstance(elem, list):
                        print(f", 元素长度={len(elem)}")
                    else:
                        print(f", 元素类型={type(elem).__name__}")
                else:
                    print()
    
    elif isinstance(first_frame, np.ndarray):
        print(f"第一帧形状: {first_frame.shape}")
        print(f"第一帧数据类型: {first_frame.dtype}")
        
        # 检查形状一致性
        print(f"\n一致性检查:")
        sample_frames = min(5, len(frames))
        shapes = [frame.shape for frame in frames[:sample_frames] if isinstance(frame, np.ndarray)]
        
        if shapes:
            unique_shapes = set(shapes)
            if len(unique_shapes) == 1:
                print(f"  前{sample_frames}帧形状一致: {shapes[0]}")
            else:
                print(f"  前{sample_frames}帧形状不一致: {list(unique_shapes)}")

def main():
    parser = argparse.ArgumentParser(description='详细查看PKL文件结构')
    parser.add_argument('--motion_file', type=str, required=True, 
                       help='动作PKL文件路径')
    parser.add_argument('--analyze_frames', action='store_true',
                       help='专门分析frames结构')
    
    args = parser.parse_args()
    
    data = inspect_pkl_file(args.motion_file)
    
    if data is not None and args.analyze_frames and isinstance(data, dict):
        analyze_frames_structure(data)

if __name__ == "__main__":
    main()