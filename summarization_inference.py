import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerFast, BartForConditionalGeneration

# KoBART base
# tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')
# model = BartForConditionalGeneration.from_pretrained('hyunwoongko/kobart')

# KoBART gogamza/kobart-summarization
# tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
# model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')

# KoBART digit82/kobart-summarization
# tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
# model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

# KoBART fine-tuning model
tokenizer = PreTrainedTokenizerFast.from_pretrained("C:\workspace\kobart_model\check_Sinmunkisa")
model = BartForConditionalGeneration.from_pretrained("C:\workspace\kobart_model\check_Sinmunkisa")

# 입력 텍스트
text = """
‘명탐정 코난’은 아오야마 고쇼 작가가 주간 소년 선데이에 30년째 연재 중인 추리 만화다. 악당 ‘검은 조직’에 의해 정신은 고등학생, 신체는 초등학생이 되어버린 고교 탐정 쿠도 신이치가 검은 조직의 정체를 파헤치며 주변에서 일어나는 미스테리한 사건들을 풀어나가는 이야기다. 만화가 인기를 끌며 TV 애니메이션, 극장판이 순차적으로 제작됐다. 국내에는 2000년 KBS에서 TV 애니메이션을 들여온 것을 시작으로 2004년부터는 투니버스에서 장기 상영됐다. 지금 20대, 30대가 돼 극장을 찾는 코난 팬들은 대부분 A씨처럼 어린 시절에 ‘더빙 코난’을 보고 자란 이들이다. A씨는 “아무래도 더빙을 오랜 기간 봤더니 더 익숙하고, 더 재밌었다”고 했다.
"""

text = text.replace('\n', ' ')


input_ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=True)

token_length = input_ids.size(1)  # 입력하는 텍스트의 사이즈

print("Token length of the input text:", token_length)

summary_ids = model.generate(
    input_ids, 
    num_beams=4, 
    max_length=128, 
    early_stopping=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    length_penalty=2.0,
    no_repeat_ngram_size=3,  
    diversity_penalty=1.0, 
    num_beam_groups=4
)

summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

print("\n",summary)