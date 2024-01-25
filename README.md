# Installation and requirements

Requires an NVIDIA GPU with CUDA support.
Uses Cuda 11.7

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117

For translations, a .env file has to be created at the root of the project with the RapidAPI SwiftTranslate API key:

RAPIDAPI_KEY="YOUR_API_KEY"

Translation can be reimplemented with another service if required.

# REST API DEFINITION

POST /conversation

| Parameter | Type | Definition | Required |
| speakers | List[Tuple[str, str]] | Defines the speakers of the conversation | TRUE |
| messages | List[Message] | List of message structures that are the conversation | TRUE |
| bark_model | str | Bark model to use. By default is suno/bark-small | FALSE |
| translateFromTo | Tuple[str, str] | Language code of the original conversation and language code to translate it to | FALSE |
| translateToSpeakers | List[Tuple[str, str]] | Speakers whose voices are going to be translated into another language |

# What it does

It creates audio conversations based on a JSON data format.
This is the data format, conversation generated by GPT-3.5.

<details>
  <summary>JSON conversation data</summary> 
  
  ```
{
    "bark_model": "suno/bark-small",
    "speakers": [
        ["Alex", "v2/en_speaker_1"],
        ["Luis", "v2/en_speaker_2"]
    ],
    "messages": [
        {
            "speaker": "Alex",
            "message": "Luis, have you ever wondered about the meaning of life, especially in this era of rapid advancements in AI?"
        },
        {
            "speaker": "Luis",
            "message": "Absolutely, Alex. It's a profound question. Do you think AI has any role in defining the purpose of our existence?"
        },
        {
            "speaker": "Alex",
            "message": "Interesting thought. While AI enhances our capabilities, I believe the meaning of life goes beyond technological advancements. What's your take?"
        },
        {
            "speaker": "Luis",
            "message": "I agree, Alex. AI may assist us, but finding purpose is a personal journey. It's about connections, experiences, and making a positive impact on the world."
        },
        {
            "speaker": "Alex",
            "message": "True. Our interactions with AI should enrich our lives, not overshadow the human experience. What values do you think are crucial in this context?"
        },
        {
            "speaker": "Luis",
            "message": "Empathy, compassion, and creativity come to mind. These human qualities define our essence and contribute to a meaningful life."
        },
        {
            "speaker": "Alex",
            "message": "Absolutely. AI can handle tasks, but the depth of human emotions and the pursuit of knowledge give life its richness. How do you see the balance between AI and humanity?"
        },
        {
            "speaker": "Luis",
            "message": "Maintaining a balance is crucial. We should leverage AI for efficiency but ensure it aligns with our values. Human connection remains irreplaceable."
        },
        {
            "speaker": "Alex",
            "message": "Well said, Luis. It's about using technology as a tool to enhance our lives rather than letting it dictate our existence. What about the ethical aspects of AI?"
        },
        {
            "speaker": "Luis",
            "message": "Ethics are vital. We need responsible AI development to prevent unintended consequences. Ensuring AI aligns with human values is key to a harmonious future."
        },
        {
            "speaker": "Alex",
            "message": "Couldn't agree more. As we navigate this AI era, fostering a global conversation on ethics and values will be crucial. What role do you see for individuals in shaping this future?"
        },
        {
            "speaker": "Luis",
            "message": "Individuals play a significant role. By staying informed, promoting ethical practices, and actively participating in discussions, we can collectively shape a positive future."
        },
        {
            "speaker": "Alex",
            "message": "Absolutely, Luis. It's a shared responsibility. As we harness the power of AI, let's ensure it aligns with our collective vision of a purposeful and meaningful life. What do you envision for the future?"
        },
        {
            "speaker": "Luis",
            "message": "I see a future where AI and humanity coexist harmoniously, complementing each other's strengths. Together, we can unlock new possibilities and make a positive impact on the world."
        },
        {
            "speaker": "Alex",
            "message": "Well said, Luis. Let's work towards that future. As we navigate this era, let's ensure the meaning of life continues to thrive, enriched by our humanity and guided by ethical principles."
        },
        {
            "speaker": "Luis",
            "message": "Agreed, Alex. It's a journey worth taking. As long as we keep our values at the forefront, the era of AI can be a force for good in defining the meaning of life."
        }
    ]
}
  ```
  
</details>

And this is the finished conversation audio

<details>
  <summary>Generated audio</summary> 

    [Audio](mp4/AI_EN.mp4)

</details>



# Translation

It can also translate that conversation into another language, using RapidAPI Swift Translate service (this can and should be reimplemented)
In order to translate, two new parameters must be added to the request, translateFromTo and translateToSpeakers.
For example:

<details>
    <summary>Added JSON for English to French</summary> 
    ```
    "translateFromTo": ["en", "fr"],
    "translateToSpeakers": [
        ["Alex", "v2/fr_speaker_1"],
        ["Luis", "v2/fr_speaker_2"]
    ]
    ```
</details>

And the resulting audio:

<details>
    <summary>French-translated conversation</summary>

    [Audio](mp4/AI_FR.mp4)
</details>




# Efficiency and timing

Bark is neither the fastest model nor the most accurate, but it's 100% free to use. Models WILL improve, and so
this little tool can be useful.

With my pathetic 3070 8GB NIVIA GPU (i will replace it soon), the small model (suno/bark-small) it takes between 6-7.5 minutes to create a 210 seconds conversation audio file (more than double the time)
The more potent model takes quite longer, a more potent GPU should dramatically increase inference times.

Also, i'm using the small model by default (bark-small). The larger model takes longer, but the quality of audio is better.

# TODO

Right now, every message is processed individually, both in terms of translation and of making bark speak the message.
According to https://huggingface.co/blog/optimizing-bark this is not the best, i'm trying to optimize so all the messages from one speaker
are processed in batch, and then reconstruct the conversation from the generated audios.
The issue is that batch processing has not been as reliable as the other one, perhaps it's something i'm doing wrong, but will keep on trying to make it run better.

Also, the translation uses a free service, when it should most likely be using a paid one. Perhaps an LLM can be useful, but i think that may just be overkill to translate a few sentences.

# Acknowledgments

To the Suno team for creating bark.
To the Hugginface team for implementing it into transformers.
To the OpenAI team for providing ChatGPT.
