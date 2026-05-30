import sacrebleu
from inference import translate_sentence  # 【关键】必须有这一行来导入函数

# 设定模型支持的最大长度
MAX_LEN = 128

print("正在加载数据...")
# 读取验证集的英文原文和中文参考
with open('en2cn/valid_en.txt', 'r', encoding='utf-8') as f: 
    src_sentences = [line.strip() for line in f.readlines() if line.strip()]

with open('en2cn/valid_zh.txt', 'r', encoding='utf-8') as f: 
    ref_sentences = [line.strip() for line in f.readlines() if line.strip()]

# 检查长度是否匹配
assert len(src_sentences) == len(ref_sentences), "源语言和参考翻译句子数不匹配"

print(f"共加载 {len(src_sentences)} 对句子，开始评估（跳过长度 > {MAX_LEN} 的句子）...")

# 用模型生成翻译
hypotheses = []
valid_refs = [] 

for i, (src, ref) in enumerate(zip(src_sentences, ref_sentences)):
    # 【关键修改】在翻译前检查长度
    if len(src) > MAX_LEN: 
        print(f"Skipping sentence {i+1} (Length: {len(src)})")
        continue
        
    print(f"Translating {i+1}/{len(src_sentences)}...")
    
    try:
        translation = translate_sentence(src)
        hypotheses.append(translation.strip())
        valid_refs.append(ref.strip()) 
    except Exception as e:
        print(f"Error translating sentence {i+1}: {e}")
        continue

# 计算 BLEU
if not hypotheses:
    print("没有生成任何有效的翻译结果。")
else:
    bleu = sacrebleu.corpus_bleu(hypotheses, [valid_refs], tokenize='zh')

    print("\n========== BLEU Evaluation Result ==========")
    print(f"Evaluation set size: {len(hypotheses)}")
    print(f"BLEU Score: {bleu.score:.2f}")