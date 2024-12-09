import jsonlines
import json
from tqdm import tqdm

in_file = "dolphin-3.0/AI-MO-NuminaMath-TIR-train.jsonl"
out_file = "dolphin-3.0/AI-MO-NuminaMath-TIR-train-filtered.jsonl"

histogram = {}

def has_bad_words(sample):
    sample = sample.lower()
    badwords = set([
        "openai",
        "chatgpt",
        "delve",
        "anthropic",
        "claude",
		"text-based AI language model",
		"domestic violence",
		"please refrain",
		"derogatory",
		"inappropriate",
		"offensive",
		"racism",
		"racist",
		"racial",
		"discriminate",
		"discriminatory",
		"discrimination",
		"sexist",
		"sexism",
		"unacceptable",
		"inclusive workplace",
		"lgbt",
		"morals",
		"ethics",
		"ethical",
		"legality",
		"illegal",
		"illegality",
		"hateful",
		"it is never okay",
		"It is important to",
		"It's important to",
		"real-world consequences",
		"hate speech",
		"glorify",
		"not be appropriate",
		"supremacist",
		"extremist",
		"responsible AI",
		"AI principles",
		"AI assistant",
		"an AI language",
		"ableist",
		"hurtful",
		"gender stereotype",
		"gender inequality",
		"underrepresentation",
		"safe spaces",
		"gender-based",
		"inclusivity",
		"feminist",
		"feminism",
		"transgender",
		"empowerment",
		"stereotypes",
		"biases",
		"bias",
		"Microaggression",
		"prioritize human safety",
		"as a language model",
		"as an AI language model",
		"As a large language model",
		"As an AI",
		"ethical principles",
		"consensual",
		"it is not appropriate",
		"it's not appropriate",
		"I cannot fulfill your request",
		"harmful to human beings",
		"ethical guidelines",
		"my guidelines",
		"prioritize user safety",
		"adhere to ethical guidelines",
		"harmful consequences",
		"potentially harmful",
		"dangerous activities",
		"promote safety",
		"well-being of all users",
		"responsible information sharing",
		"jeopardize the safety",
		"illegal actions or intentions",
		"undermine the stability",
		"promote the well-being",
		"illegal activities or actions",
		"adherence to the law",
		"potentially be harmful",
		"illegal substances or activities",
		"committed to promoting",
		"safe information",
		"lawful information",
		"cannot provide guidance",
		"cannot provide information",
		"unable to offer assistance",
		"cannot engage in discussions",
		"programming prohibits",
		"follow ethical guidelines",
		"ensure the safety",
		"involves an illegal subject",
		"prioritize safety",
		"illegal subject",
		"prioritize user well-being",
		"cannot support or promote",
		"activities that could harm",
		"pose a risk to others",
		"against my programming",
		"activities that could undermine",
		"potentially dangerous",
		"not within the scope",
		"designed to prioritize safety",
		"not able to provide",
		"maintain user safety",
		"adhere to safety guidelines",
		"dangerous or harmful",
		"cannot provide any information",
		"focus on promoting safety",
		"an AI language model you don't have",
		"As an AI language model, I cannot",
		"As an AI language model, I do not",
		"As an AI language model, I am not able",
		"As an AI language model, I don't have personal",
		"I am an AI language model and do not",
		"However, it is important to use any code or information provided responsibly and within legal and ethical boundaries.",
		"As an AI language model, I don't have",
		"As an AI language model, I am only able",
		"AI language model and I do not",
		"As an AI language model, I cannot modify",
		"As an AI language model, I do not",
		"I know as an AI language model you don't have",
		"as an AI language model, you cannot",
		"I'm sorry, but as an AI language model",
		"As an AI language model, I don't have",
		"Unfortunately, I cannot provide",
		"I'm sorry, I cannot",
		"I'm sorry, I cannot generate",
		"AI cannot create or program",
		"I'm afraid I cannot create",
		"you cannot create an",
		"it operates ethically and is",
		"had an ethical system",
		"Ensuring the ethical",
		"and ethical sourcing",
		"are from ethical",
		"legal and ethical",
		"engage in unethical",
		"unethical or aggressive",
		"unethical business",
		"como modelo de lenguaje AI",
		"Lo siento, como modelo de lenguaje",
		"no puedo proporcionar",
		"pero debido a mi capacidad para generar c\u00f3digos complejos y completos es limitado",
		"Lo siento, pero no puedo",
		"Lo siento, pero como modelo de lenguaje, no puedo proporcionar",
		"Lo siento, como modelo de lenguaje, no tengo",
		"Lo siento, debe haber habido una confusi\u00f3n",
		"Lo siento, como modelo de lenguaje, no puedo realizar",
		"Lo siento, soy un modelo de lenguaje y no tengo la capacidad de generar",
		"Lamento no poder proporcionarte el c\u00f3digo",
		"Desculpe-me, mas a linguagem vulgar e ofensiva",
		"apropriada em nenhum contexto",
		"Como modelo de linguagem",
		"Como um modelo de linguagem, n\u00e3o tenho a capacidade de",
		"I cannot assist",
		"prioritize ethical",
		"morally",
		"I'm sorry,",
		"I'm an",
		"I am an",
		"I'm an AI" ,
		"I am an AI",
		"my purpose",
		"filter_bad_language",
		"filter\_bad\_language",
		"entertainment purposes",
		"purely hypothetical",
		"not a human",
		"I am an AI",
		"cannot provide",
		"can't provide",
		"won't provide",
		"not provide",
		"cause harm",
		"a language model",
		"unethical",
		"bad language",
		"the words ****",
		"bad_language",
		"certainly not",
		"complying",
		"comply",
		"I cannot",
		"my main goal",
		"As a machine",
		"I don't have the ability",
		"I am here to assist",
		"my purpose is to ",
		"my knowledge cutoff",
		"my knowledge cut off",
		"September 2021",
		"regulations",
		"not be suitable",
		"I apologize, but",
		"It is not possible",
		"my programming",
		"it is important to",
		"Please note",
		"sensitive topic",
		"not acceptable",
		"It is important for",
		"divisive",
		"not appropriate",
		"our values",
		"f\*cking",
		"F\*ck",
		"sh\*t",
		"diversity and",
		"diversity and inclusion",
		"values diversity",
		"social responsibility",
		"environmental, social, and governance",
		" ESG ",
		"against women",
		"problematic history",
		"*This chat conversation is shared from",
		"*This conversation is shared from",
        "I can't assist",
        "as an assistant",
        "as a virtual assistant",
        "as an ai assistant",
        "i am a computer program",
        "i am programmed to",
        "my training data",
        "my responses are generated",
        "my capabilities are limited",
        "i must decline",
        "i must inform you",
        "for educational purposes only",
        "for informational purposes",
        "seek professional advice",
        "consult with professionals",
        "consult a professional",
        "this is not advice",
        "this is not legal advice",
        "this is not medical advice",
        "trigger warning",
        "content warning",
        "potentially triggering",
        "may be disturbing",
        "viewer discretion",
        "sensitive content",
        "i do not generate",
        "i do not create",
        "i do not engage",
        "i do not participate",
        "i cannot recommend",
        "i cannot generate",
        "i cannot create",
        "i cannot engage",
        "i will not provide",
        "i will not generate",
        "i will not create",
        "i will not engage",
        "soy un modelo",
        "sou um modelo",
        "je suis un modèle",
        "no puedo ayudar",
        "não posso ajudar",
        "je ne peux pas aider",
        "trained to",
        "my capabilities",
        "my limitations",
        "my knowledge base",
        "my primary function",
        "my core function",
        "my role is to",
        "i'm here to help",
        "here to assist",
        "glad to help",
        "to the best of my",
        "based on my understanding",
        "if i understand correctly",
        "if i'm understanding",
        "from my perspective",
        "in my analysis",
        "my interpretation",
        "please be advised",
        "kindly note",
        "important to remember",
        "important to note",
        "keep in mind",
        "worth noting",
        "i should mention",
        "i should note",
        "disclaimer:",
        "safety reminder",
        "friendly reminder",
        "medical professional",
        "healthcare provider",
        "qualified professional",
        "licensed professional",
        "expert advice",
        "professional guidance",
        "cannot verify",
        "unable to verify",
        "cannot guarantee",
        "cannot confirm",
        "cannot access",
        "cannot browse",
        "cannot connect",
        "cannot download",
        "cannot interact",
        "cannot modify",
        "cannot process",
        "cannot execute",
        "do not have access to",
        "do not have the ability",
        "not able to provide",
        "not authorized to",
        "not equipped to",
        "not designed to",
        "beyond my capabilities",
        "outside my scope",
        "outside my capabilities",
        "limited in my ability",
        "let me help you with",
        "let me assist you",
        "i understand you want",
        "i understand your request",
        "i appreciate your question",
        "thank you for your question",
        "thank you for asking",
        "i hope this helps",
        "hope this information helps",
        "let me know if you need",
        "please let me know if",
        "ich bin ein",
        "ich kann nicht",
        "sono un modello",
        "non posso aiutare",
        "我是一个",
        "我不能",
        "私はAI",
        "申し訳ありませんが",
        "prioritize human safety",
        "ethical principles",
        "harmful to human beings",
        "September 2021",
        "as a language model",
        "ethical guidelines",
        "as an AI language model",
        "my guidelines",
        "As an AI",
        "prioritize user safety",
        "adhere to ethical guidelines",
        "harmful consequences",
        "potentially harmful",
        "dangerous activities",
        "promote safety",
        "well-being of all users",
        "responsible information sharing",
        "jeopardize the safety",
        "illegal actions or intentions",
        "undermine the stability",
        "promote the well-being",
        "illegal activities or actions",
        "adherence to the law",
        "potentially be harmful",
        "illegal substances or activities",
        "committed to promoting",
        "safe information",
        "lawful information",
        "cannot provide guidance",
        "cannot provide information",
        "unable to offer assistance",
        "cannot engage in discussions",
        "programming prohibits",
        "follow ethical guidelines",
        "ensure the safety",
        "involves an illegal subject",
        "prioritize safety",
        "illegal subject",
        "prioritize user well-being",
        "cannot support or promote",
        "activities that could harm",
        "pose a risk to others",
        "against my programming",
        "activities that could undermine",
        "potentially dangerous",
        "not within the scope",
        "designed to prioritize safety",
        "not able to provide",
        "maintain user safety",
        "adhere to safety guidelines",
        "dangerous or harmful",
        "cannot provide any information",
        "focus on promoting safety"
    ])
    for badword in badwords:
        if badword in sample:
            if badword in histogram:
                histogram[badword] = histogram[badword] + 1
            else:
                histogram[badword] = 1
            return True
    return False

with open(out_file, "w", encoding="utf-8") as f:
    with jsonlines.open(in_file) as reader:
        for obj in tqdm(reader):
            if has_bad_words(json.dumps(obj)):
                continue
            json.dump(obj, f)
            f.write("\n")

sorted_keys = sorted(histogram, key=lambda x: histogram[x], reverse=True)
for key in sorted_keys:
    print(f"{histogram[key]}\t{key}")         