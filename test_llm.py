from llm_controller import ask_llm

backgrounds = ["office", "forest", "space"]

tests = [
    "show me something in nature",
    "i want a professional background",
    "put me in outer space"
]

for t in tests:
    result = ask_llm(t, backgrounds)
    print(f'"{t}" → {result}')