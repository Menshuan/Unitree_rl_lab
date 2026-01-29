import argparse
import sys
import numpy as np

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Check contents of an NPZ file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py --motion_file data.npz
  python script.py -f motions/example.npz --brief
  python script.py -f data.npz --save-keys keys.txt
        """
    )
    
    # 添加参数
    parser.add_argument(
        '-f', '--motion_file',
        type=str,
        required=True,
        help='Path to the NPZ file to analyze'
    )
    
    parser.add_argument(
        '-b', '--brief',
        action='store_true',
        help='Only show keys, not detailed info'
    )
    
    parser.add_argument(
        '-s', '--save-keys',
        type=str,
        help='Save keys to a text file'
    )
    
    parser.add_argument(
        '--allow-pickle',
        action='store_true',
        help='Allow loading pickled objects (use with caution)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show more detailed information'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    try:
        # 加载npz文件
        if args.allow_pickle:
            data = np.load(args.motion_file, allow_pickle=True)
        else:
            data = np.load(args.motion_file)
        
        print(f"✓ Successfully loaded: {args.motion_file}")
        print(f"📁 Total arrays: {len(data.files)}")
        print("=" * 50)
        
        # 打印键名
        keys = list(data.files)
        print("Keys in npz file:", keys)
        
        # 如果指定了保存keys到文件
        if args.save_keys:
            with open(args.save_keys, 'w') as f:
                for key in keys:
                    f.write(f"{key}\n")
            print(f"✓ Keys saved to: {args.save_keys}")
        
        # 如果不是brief模式，显示详细信息
        if not args.brief:
            print("\n" + "=" * 50)
            print("DETAILED INFORMATION:")
            print("=" * 50)
            
            for key in keys:
                array = data[key]
                print(f"\n🔹 Key: {key}")
                print(f"   Shape: {array.shape}")
                print(f"   Dtype: {array.dtype}")
                print(f"   Size: {array.size:,} elements")
                print(f"   Dimensions: {array.ndim}D")
                print(f"   Memory: {array.nbytes:,} bytes ({array.nbytes/1024/1024:.2f} MB)")
                
                # 如果是verbose模式，显示更多信息
                if args.verbose:
                    if np.issubdtype(array.dtype, np.number):
                        print(f"   Min: {array.min():.6f}")
                        print(f"   Max: {array.max():.6f}")
                        print(f"   Mean: {array.mean():.6f}")
                        if array.size > 1:
                            print(f"   Std: {array.std():.6f}")
                    
                    # 显示前几个元素（如果是小数组或1D数组）
                    if array.ndim == 1 and array.size <= 10:
                        print(f"   Values: {array}")
                    elif array.ndim == 1 and array.size > 10:
                        print(f"   First 5 values: {array[:5]} ...")
                    elif array.ndim == 2 and array.shape[0] <= 5 and array.shape[1] <= 5:
                        print(f"   Matrix:\n{array}")
        
        # 计算总大小
        total_bytes = sum(data[key].nbytes for key in keys if isinstance(data[key], np.ndarray))
        print(f"\n📊 Total file size: {total_bytes/1024/1024:.2f} MB")
        
        # 关闭文件
        data.close()
        
    except FileNotFoundError:
        print(f"❌ Error: File '{args.motion_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading NPZ file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()