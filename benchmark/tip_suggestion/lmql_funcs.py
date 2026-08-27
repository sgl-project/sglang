number = 5


async def expand_tip_async(topic, tip, generate):
    s = (
        """Please expand a tip for a topic into a detailed paragraph.

Topic: staying healthy
Tip: Regular Exercise
Paragraph: Incorporate physical activity into your daily routine. This doesn't necessarily mean intense gym workouts; it can be as simple as walking, cycling, or yoga. Regular exercise helps in maintaining a healthy weight, improves cardiovascular health, boosts mental health, and can enhance cognitive function, which is crucial for fields that require intense intellectual engagement.

Topic: building a campfire
Tip: Choose the Right Location
Paragraph: Always build your campfire in a safe spot. This means selecting a location that's away from trees, bushes, and other flammable materials. Ideally, use a fire ring if available. If you're building a fire pit, it should be on bare soil or on a bed of stones, not on grass or near roots which can catch fire underground. Make sure the area above is clear of low-hanging branches.

Topic: writing a blog post
Tip: structure your content effectively
Paragraph: A well-structured post is easier to read and more enjoyable. Start with an engaging introduction that hooks the reader and clearly states the purpose of your post. Use headings and subheadings to break up the text and guide readers through your content. Bullet points and numbered lists can make information more digestible. Ensure each paragraph flows logically into the next, and conclude with a summary or call-to-action that encourages reader engagement.

Topic: """
        + topic
        + "\nTip: "
        + tip
        + "\nParagraph:"
    )
    return await generate(s, max_tokens=128, stop="\n\n")


async def suggest_tips_async(topic, generate):
    s = "Please act as a helpful assistant. Your job is to provide users with useful tips on a specific topic.\n"
    s += "USER: Give some tips for " + topic + ".\n"
    s += (
        "ASSISTANT: Okay. Here are "
        + str(number)
        + " concise tips, each under 8 words:\n"
    )

    tips = []
    for i in range(1, 1 + number):
        s += f"{i}."
        # NOTE: stop is different due to lmql does not support a list of stop tokens
        tip = await generate(s, max_tokens=24, stop=".\n")
        s += tip + ".\n"
        tips.append(tip)

    paragraphs = [await expand_tip_async(topic, tip, generate=generate) for tip in tips]

    for i in range(1, 1 + number):
        s += f"Tip {i}:" + paragraphs[i - 1] + "\n"
    return s
