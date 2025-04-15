# -*- coding: utf-8 -*-
import subprocess
import argparse
import os
import sys
import time

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LSTM Sentiment Analysis System')
    parser.add_argument('--skip_preprocessing', action='store_true', help='Skip data preprocessing step')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training step')
    parser.add_argument('--sample_size', type=int, default=None, 
                        help='Sample size to use, default is None (all data). Set a value like 5000 for quick testing.')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs, default is 3')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training, default is 128')
    parser.add_argument('--max_vocab_size', type=int, default=30000,
                        help='Maximum vocabulary size, default is 30000')
    parser.add_argument('--max_length', type=int, default=200,
                        help='Maximum sequence length, default is 200')
    
    args = parser.parse_args()
    
    # 数据预处理
    if not args.skip_preprocessing:
        print("=" * 50)
        print("Step 1: Data Preprocessing")
        print("=" * 50)
        try:
            # 构建预处理命令
            preprocess_cmd = [sys.executable, 'data_preprocessing.py', 
                             '--max_length', str(args.max_length),
                             '--max_vocab_size', str(args.max_vocab_size)]
            
            # 如果指定了样本大小，则添加到命令中
            if args.sample_size is not None:
                preprocess_cmd.extend(['--sample_size', str(args.sample_size)])
                
            # 执行预处理
            subprocess.run(preprocess_cmd, check=True)
            print("Data preprocessing complete")
        except subprocess.CalledProcessError as e:
            print(f"Data preprocessing failed: {e}")
            sys.exit(1)
    else:
        print("Skipping data preprocessing step...")
    
    # 模型训练
    if not args.skip_training:
        print("\n" + "=" * 50)
        print("Step 2: Model Training")
        print("=" * 50)
        try:
            # 构建训练命令
            train_cmd = [sys.executable, 'model.py',
                        '--epochs', str(args.epochs),
                        '--batch_size', str(args.batch_size)]
            
            # 执行训练
            subprocess.run(train_cmd, check=True)
            print("Model training complete")
        except subprocess.CalledProcessError as e:
            print(f"Model training failed: {e}")
            sys.exit(1)
    else:
        print("Skipping model training step...")
    
    # 启动Web应用
    print("\n" + "=" * 50)
    print("Step 3: Starting Web Application")
    print("=" * 50)
    print("Web application is starting, please visit http://localhost:5000")
    try:
        # 对于Web应用，我们不使用check=True，因为用户可能会手动中断
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\nWeb application stopped by user")
    except Exception as e:
        print(f"Web application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 打印内存使用优化提示
    if os.name == 'posix':  # Linux/Mac
        print("Memory usage optimization tip: You can set environment variables for better memory management:")
        print("export PYTHONUNBUFFERED=1")
        print("export OMP_NUM_THREADS=4")
    
    start_time = time.time()
    main()
    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.2f} minutes") 