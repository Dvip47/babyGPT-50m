import random

# Sample data for each category
code_examples = [
    ("# Add two numbers", "def add(a, b):\n    return a + b"),
    ("# Check if a number is even", "def is_even(n):\n    return n % 2 == 0"),
    ("# Find maximum of two numbers", "def max_num(a, b):\n    return a if a > b else b"),
    ("# Calculate factorial", "def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)"),
    ("# Check for prime number", "def is_prime(n):\n    return all(n % i != 0 for i in range(2, n)) if n > 1 else False"),
]

en_gk = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote 'Hamlet'?", "William Shakespeare"),
    ("Which planet is called the Red Planet?", "Mars"),
    ("What is the boiling point of water?", "100°C"),
    ("What is the currency of Japan?", "Yen"),
]

hi_gk = [
    ("भारत का राष्ट्रीय पक्षी कौन है?", "मोर"),
    ("गांधी जयंती कब मनाई जाती है?", "2 अक्टूबर"),
    ("भारत के पहले प्रधानमंत्री कौन थे?", "पंडित जवाहरलाल नेहरू"),
    ("ताजमहल कहाँ स्थित है?", "आगरा"),
    ("हिमालय किस दिशा में स्थित है?", "उत्तर"),
]

# Function to generate a block
def create_block():
    block_type = random.choice(['CODE', 'EN_GK', 'HI_GK'])
    if block_type == 'CODE':
        title, code = random.choice(code_examples)
        return f"[CODE]\n{title}\n{code}\n"
    elif block_type == 'EN_GK':
        q, a = random.choice(en_gk)
        return f"[EN_GK]\nQ: {q}\nA: {a}\n"
    elif block_type == 'HI_GK':
        q, a = random.choice(hi_gk)
        return f"[HI_GK]\nQ: {q}\nA: {a}\n"

# Generate 1000 blocks and write to input.txt
with open("data/multitask/input.txt", "w", encoding="utf-8") as f:
    for _ in range(1000):
        f.write(create_block() + "\n")

print("✅ input.txt created with 1000 diverse examples!")
