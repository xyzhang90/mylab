import os

def split_tsv(tsv_file):
    # 检查文件是否存在
    if not os.path.exists(tsv_file):
        print(f"❌ 错误：找不到文件 {tsv_file}")
        return

    # 输出目录设为当前目录下的 en2cn
    output_dir = 'en2cn'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"🚀 开始处理: {tsv_file}")
    
    count_total = 0
    count_valid = 0
    
    with open(tsv_file, 'r', encoding='utf-8') as f_in, \
         open(os.path.join(output_dir, 'train_en.txt'), 'w', encoding='utf-8') as f_en, \
         open(os.path.join(output_dir, 'train_zh.txt'), 'w', encoding='utf-8') as f_zh:
        
        for i, line in enumerate(f_in):
            count_total += 1
            
            # 仅打印前3行用于调试，看看数据长什么样
            if i < 3:
                print(f"  [调试] 第{i+1}行原始内容: {repr(line)}")

            line = line.strip()
            if not line: continue
            
            parts = line.split('\t')
            
            # 必须至少有2列才能处理
            if len(parts) >= 2:
                f_en.write(parts[0].strip() + '\n')
                f_zh.write(parts[1].strip() + '\n')
                count_valid += 1

    print("-" * 30)
    print(f"📊 统计结果:")
    print(f"   总行数读取: {count_total}")
    print(f"   有效双语行: {count_valid}")
    
    if count_valid == 0:
        print("⚠️ 警告：没有提取到任何数据！请检查TSV文件格式是否包含制表符(Tab)。")
    else:
        print(f"✅ 成功生成文件在 ./{output_dir}/ 目录下")

if __name__ == '__main__':
    split_tsv('raw_data.tsv')