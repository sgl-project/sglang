import sglang as sgl

# here are the top five agent functions contributing ~70% LLM calls
# reference: https://github.com/joonspk-research/generative_agents/


@sgl.function
def poignancy_event(s, persona_name, persona_iss, event):
    s += "Here is a brief description of " + persona_name + ".\n"
    s += persona_iss + "\n"
    s += "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following event for"
    s += persona_name + ".\n\n"
    s += "Event: " + event
    s += "Rate (return a number between 1 to 10):"
    s += sgl.gen(name="Rate", max_tokens=2)


def poignancy_event_prompt(persona_name, persona_iss, event):
    # return prompt and max_tokens
    s = ""
    s += "Here is a brief description of " + persona_name + ".\n"
    s += persona_iss + "\n"
    s += "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following event for"
    s += persona_name + ".\n\n"
    s += "Event: " + event
    s += "Rate (return a number between 1 to 10):"
    return {"prompt": s, "max_tokens": 2, "stop": None}


@sgl.function
def generate_event_triple(s, persona_name, action):
    s += """Task: Turn the input into (subject, predicate, object).
Input: Sam Johnson is eating breakfast.
Output: (Dolores Murphy, eat, breakfast)
---
Input: Joon Park is brewing coffee.
Output: (Joon Park, brew, coffee)
---
Input: Jane Cook is sleeping.
Output: (Jane Cook, is, sleep)
---
Input: Michael Bernstein is writing email on a computer.
Output: (Michael Bernstein, write, email)
---
Input: Percy Liang is teaching students in a classroom.
Output: (Percy Liang, teach, students)
---
Input: Merrie Morris is running on a treadmill.
Output: (Merrie Morris, run, treadmill)
---"""
    s += persona_name + "is" + action + ".\n"
    s += "(" + persona_name + ","
    s += sgl.gen(name="Triple", max_tokens=20, stop=")")


def generate_event_triple_prompt(persona_name, action):
    s = ""
    s += """Task: Turn the input into (subject, predicate, object).
Input: Sam Johnson is eating breakfast.
Output: (Dolores Murphy, eat, breakfast)
---
Input: Joon Park is brewing coffee.
Output: (Joon Park, brew, coffee)
---
Input: Jane Cook is sleeping.
Output: (Jane Cook, is, sleep)
---
Input: Michael Bernstein is writing email on a computer.
Output: (Michael Bernstein, write, email)
---
Input: Percy Liang is teaching students in a classroom.
Output: (Percy Liang, teach, students)
---
Input: Merrie Morris is running on a treadmill.
Output: (Merrie Morris, run, treadmill)
---"""
    s += persona_name + "is" + action + ".\n"
    s += "(" + persona_name + ","
    return {"prompt": s, "max_tokens": 20, "stop": ")"}


@sgl.function
def generate_pronunciatio(s, action):
    s += "Convert an action description to an emoji (important: use two or less emojis).\n"
    s += "Action description: " + action + ".\n"
    s += "Emoji:" + sgl.gen(name="Emoji", max_tokens=6)


def generate_pronunciatio_prompt(action):
    s = ""
    s += "Convert an action description to an emoji (important: use two or less emojis).\n"
    s += "Action description: " + action + ".\n"
    s += "Emoji:"
    return {"prompt": s, "max_tokens": 6, "stop": None}


@sgl.function
def action_location_sector(
    s,
    persona_name,
    living_sector,
    living_sector_areas,
    current_sector,
    current_sector_areas,
    daily_plan,
    sector_options,
    current_action,
    next_action,
):
    s += """Task -- choose an appropriate area  from the area options for a task at hand.
Sam Kim lives in {Sam Kim's house} that has Sam Kim's room, bathroom, kitchen.
Sam Kim is currently in {Sam Kim's house} that has Sam Kim's room, bathroom, kitchen.
Area options: {Sam Kim's house, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy}.
* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the "Area options," verbatim.
For taking a walk, Sam Kim should go to the following area: {Johnson Park}
---
Jane Anderson lives in {Oak Hill College Student Dormatory} that has Jane Anderson's room.
Jane Anderson is currently in {Oak Hill College} that has a classroom, library
Area options: {Oak Hill College Student Dormatory, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy}.
* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the "Area options," verbatim.
For eating dinner, Jane Anderson should go to the following area: {Hobbs Cafe}
---"""
    s += (
        persona_name
        + " lives in "
        + living_sector
        + " that has "
        + living_sector_areas
        + ".\n"
    )
    s += (
        persona_name
        + " is currently in "
        + current_sector
        + " that has "
        + current_sector_areas
        + ".\n"
    )
    s += daily_plan + ".\n"
    s += "Area options: " + sector_options + ".\n"
    s += """* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the "Area options," verbatim.\n"""
    s += (
        persona_name
        + " is "
        + current_action
        + ". For "
        + next_action
        + ", "
        + persona_name
        + " should go to the following area: {"
    )
    s += sgl.gen(name="Location", max_tokens=10, stop="}")


def action_location_sector_prompt(
    persona_name,
    living_sector,
    living_sector_areas,
    current_sector,
    current_sector_areas,
    daily_plan,
    sector_options,
    current_action,
    next_action,
):
    s = ""
    s += """Task -- choose an appropriate area  from the area options for a task at hand.
Sam Kim lives in {Sam Kim's house} that has Sam Kim's room, bathroom, kitchen.
Sam Kim is currently in {Sam Kim's house} that has Sam Kim's room, bathroom, kitchen.
Area options: {Sam Kim's house, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy}.
* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the "Area options," verbatim.
For taking a walk, Sam Kim should go to the following area: {Johnson Park}
---
Jane Anderson lives in {Oak Hill College Student Dormatory} that has Jane Anderson's room.
Jane Anderson is currently in {Oak Hill College} that has a classroom, library
Area options: {Oak Hill College Student Dormatory, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy}.
* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the "Area options," verbatim.
For eating dinner, Jane Anderson should go to the following area: {Hobbs Cafe}
---"""
    s += (
        persona_name
        + " lives in "
        + living_sector
        + " that has "
        + living_sector_areas
        + ".\n"
    )
    s += (
        persona_name
        + " is currently in "
        + current_sector
        + " that has "
        + current_sector_areas
        + ".\n"
    )
    s += daily_plan + ".\n"
    s += "Area options: " + sector_options + ".\n"
    s += """* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the "Area options," verbatim.\n"""
    s += (
        persona_name
        + " is "
        + current_action
        + ". For "
        + next_action
        + ", "
        + persona_name
        + " should go to the following area: {"
    )
    return {"prompt": s, "max_tokens": 10, "stop": "}"}


@sgl.function
def action_location_object(
    s, persona_name, target_sector, target_sector_areas, current_action, next_action
):
    s += """
Jane Anderson is in kitchen in Jane Anderson's house.
Jane Anderson is going to Jane Anderson's house that has the following areas: {kitchen,  bedroom, bathroom}
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For cooking, Jane Anderson should go to the following area in Jane Anderson's house:
Answer: {kitchen}
---
Tom Watson is in common room in Tom Watson's apartment.
Tom Watson is going to Hobbs Cafe that has the following areas: {cafe}
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For getting coffee, Tom Watson should go to the following area in Hobbs Cafe:
Answer: {cafe}
---"""
    s += (
        persona_name
        + " is going to "
        + target_sector
        + " that has the following areas: {"
        + target_sector_areas
        + "}\n"
    )
    s += """* Stay in the current area if the activity can be done there.
* NEVER go into other people's rooms unless necessary."""
    s += (
        persona_name
        + " is "
        + current_action
        + ". For "
        + next_action
        + ", "
        + persona_name
        + "should go to the following area in "
        + target_sector
    )
    s += " (MUST pick one of {" + target_sector_areas + "}):\n"
    s += "Answer: {" + sgl.gen(name="Area", max_tokens=5, stop="}")


def action_location_object_prompt(
    persona_name, target_sector, target_sector_areas, current_action, next_action
):
    s = ""
    s += """
Jane Anderson is in kitchen in Jane Anderson's house.
Jane Anderson is going to Jane Anderson's house that has the following areas: {kitchen,  bedroom, bathroom}
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For cooking, Jane Anderson should go to the following area in Jane Anderson's house:
Answer: {kitchen}
---
Tom Watson is in common room in Tom Watson's apartment.
Tom Watson is going to Hobbs Cafe that has the following areas: {cafe}
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For getting coffee, Tom Watson should go to the following area in Hobbs Cafe:
Answer: {cafe}
---"""
    s += (
        persona_name
        + " is going to "
        + target_sector
        + " that has the following areas: {"
        + target_sector_areas
        + "}\n"
    )
    s += """* Stay in the current area if the activity can be done there.
* NEVER go into other people's rooms unless necessary."""
    s += (
        persona_name
        + " is "
        + current_action
        + ". For "
        + next_action
        + ", "
        + persona_name
        + "should go to the following area in "
        + target_sector
    )
    s += " (MUST pick one of {" + target_sector_areas + "}):\n"
    s += "Answer: {"
    return {"prompt": s, "max_tokens": 5, "stop": "}"}
