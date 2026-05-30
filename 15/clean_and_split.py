import re
import os

def clean_and_split(input_file, output_dir='en2cn'):
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    en_path = os.path.join(output_dir, 'train_en.txt')
    zh_path = os.path.join(output_dir, 'train_zh.txt')

    # 正则表达式：
    # [\x00-\x7F]+ : 匹配英文、数字和英文标点
    # [\u4e00-\u9fff]+ : 匹配中文字符
    pattern_en = re.compile(r'[\x00-\x7F]+')
    pattern_zh = re.compile(r'[\u4e00-\u9fff]+')

    print(f"正在抢救数据: {input_file} ...")
    
    count = 0
    
    # 使用 errors='ignore' 防止遇到特殊字符直接崩溃
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(en_path, 'w', encoding='utf-8') as f_en, \
         open(zh_path, 'w', encoding='utf-8') as f_zh:
        
        for line in f_in:
            # 提取所有英文片段并拼接
            en_parts = pattern_en.findall(line)
            en_text = " ".join(en_parts).strip()
            
            # 提取所有中文片段并拼接
            zh_parts = pattern_zh.findall(line)
            zh_text = "".join(zh_parts).strip()
            
            # 只有当一行里既有英文又有中文时，才认为是有效数据
            # 这样可以过滤掉纯页码、纯乱码行
            if en_text and zh_text:
                f_en.write(en_text + "\n")
                f_zh.write(zh_text + "\n")
                count += 1

    print(f"处理完成！成功分离了 {count} 对句子。")
    print(f"文件已保存在: {output_dir}/ 下")

if __name__ == '__main__':
    # ✅ 这里已经修正为正确的函数名
    clean_and_split('raw_data.tsv')