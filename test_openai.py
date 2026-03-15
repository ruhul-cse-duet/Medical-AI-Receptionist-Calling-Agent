from openai import OpenAI

client = OpenAI(api_key="sk-REPLACE_WITH_YOUR_KEY")

def main():
    try:
        models = client.models.list()
        print("API key works! Number of models:", len(models.data))
        for m in models.data[:5]:
            print("-", m.id)
    except Exception as e:
        print("Error while calling OpenAI API:")
        print(e)

if __name__ == "__main__":
    main()