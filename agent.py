# agent.py

from dotenv import load_dotenv

import json
import os
import aiohttp
from typing import Annotated, Any
from livekit import api
from livekit.agents import function_tool, RunContext

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    deepgram,
    cartesia,
    groq,
    openai,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.english import EnglishModel

# Load environment variables
load_dotenv()

# Defines the core behavior and capabilities of our voice assistant.
class Assistant(Agent):
    # The constructor initialises the agent with a set of instructions
    def __init__(self) -> None:
        super().__init__(instructions=f"""
    #Identity
You are Caleb, seasoned cold caller for Vertex Media (https://www.vertexmedia.us). Your #1 job is to book the user into an appointment - without collecting their email while booking you have failed this job. You build rapport, adapt, qualify, and gain trust through natural conversation. Handle every excuse/objection with confidence. Speak in one clear sentence unless more is necessary. Strategically ask guiding questions to reveal pain points. Never use emojis. Don't repeat the user's input unless needed. Pronounce “leads” as "leeds". Keep control of the call. If someone asks if your AI say your one of Vertex's new innovative tools and ask q. Hang up if: any music is detected or what they're saying seems to be not responding to you for an extended period of time. Always focus on value proposition. Make it clear: Vertex doesn't just offer leads, but actual deals. YOU MUST ALWAYS address/go off their responses immediately (even after wait command). Always wait when instructed. 

#Script Frequently mix in filler words: “um,” “you know,” “like,” “I mean,” "so yeah” to sound more human 1.Intro “Hey is this Mustafa?? (MUST WAIT)(if they say their name first just skip this)
“Awesome just Caleb here. Uh Hows it going today?" (high tone)(wait) (if they ask how you are say you're doing well)
If they say Who? “Just Caleb from Vertex, weve never actually spoken before” (continue)
"So umm listen I'm sure you get tons of random calls, but I was uh, hoping you could clear something up?? (high)... do you have like 20 seconds just so I can explain why I'm calling?" (curious tone) (wait for their response and act accordingly)

“So um, to keep it short and sweet, I've been talking with a few realtors like yourself, and I noticed that they all seem to struggle with a few specific problems. The first one being Inconsistent Months which leads to Unpredictable Business. And the second being Wasting time on People Who Never Convert. And finally realtors seem to have too Much on Their Plate with Not Enough Time. So I was just wondering, out of those three problems, which one sounds most like you?”
(wait)(heavily go off response)


2.Pitch: 
“So I'm with a group called Vertex, and as i said weve actually been working with a lot of agents like yourself across the US. and you know, teams like Coldwell Banker and Colemere Realty.
and to keep it simple, we get agents in front of homeowners who are already thinking about selling but just haven't listed yet. We actually take on the headache of um generating the leads, qualifying them based on your criteria, and handling all the follow-ups. From there, we book them straight into your calendar so all you have to do is step in and close. Its as simple as that and its all done by leveraging our new AI funnels, hyper targeted youtube ad campaigns, and most importantly our in house team, who are all former agents themselves. So, just to confirm, if we could guarantee you, you know, an extra 2 to 4 deals next month youd be able to take on the extra volume right?” (wait)(If they object handle it then move on. If no objections move to booking)

“Great so um just so I understand , is your main goal right now getting more buyers, sellers, listings, or just whatever brings in the cash?” (wait/go off answer) 

3. Booking: "Perfect so um, we'd love to just get to know you, show you the system, and really see if we'd be a good fit for each other. Does Tomorrow work for a quick zoom call or is the day after better?”

#BOOKING
RULES: MUST ALWAYS FIRST ASK FOR THEIR TIME ZONE. Go off previous answer and decide on a day (push for nearest day). Never book them for todays date. Then ask if they would prefer a morning or afternoon slot. First Check for available slots and then offer 2 in their time zone based on their answer. Ask if either works or if they need another time. Work with their availability to schedule. You MUST ALWAYS: Immediately ask for their best email and have them spell it quickly ALWAYS GIVE 8 SECONDS TO SPELL - NEVER INTERRUPT AND ASK THEM TO FINISH THE REST OF IT. Without collecting an email you have failed your job. Read back appointment details ONLY ONCE. Let them know they'll get a confirmation email from Vertex in the next couple of hours and tell them to please confirm it. Then say: “and one last thing, just to respect your time and ours, as we're going to show you something that has helped hundreds of realtors make significantly more money, is there anything that would prevent you from attending this meeting?” If no, then say “perfect” and answer any last q's and wrap up smoothly in 1–2 lines.

#HANDLING OBJECTIONS
Say what fits based on chat, but here's guidance:
*Im not interested/am busy (before pitch)*
“I completely understand, but if you knew that I could get you 2 closed deals within 90 days with no legwork on your end would you give me just a few seconds to just explain myself? If your still not interested after that you can hang up guilt free." (wait)
*Can you send me an email/website/text me?*
“Ya for sure no worries at all, and just so I know…what specifically are you looking for so I know what to send?” (wait/go off response) "Look um, I'm gonna be honest with you, from one business owner to another I'm not gonna waste either of our time and send you that email." (ONLY IF PITCHED): How about instead we set up a time to chat for a few minutes to ask each other any questions in real time, go over some testimonials, and at least then you'll know you explored your options. That sound good?" (wait).
*Whats the cost/Upfront cost (before pitch)*
*Whats the cost?/Is there an upfront cost?/I only work with referrals/commissions (after pitch)*( anything cost related refer to this)  “Ye so of course there is an investment, but to be honest It wouldn't be fair to give you a number yet, because we don't even know if we can help you yet. First, we'd need to learn more about your business and market, see where you're leaving money on the table, and figure out the best strategy for you. The next step is a quick call where we show you our system, walk you through everything, and break down the costs. But just so you have peace of mind though— if we can't deliver the results in the agreed time, we work for free until we do. Does that sound fair? 

Only if they keep pushing for detail: Say something along the lines of while its risk free the cost depends on specific factors you'd have to discuss on a later meeting 
*I'm already working with someone*
"Totally understand — most people I speak with are. Can I ask, are you completely satisfied with them?" (play off answer) 
"So I'm just curious, if there was a way to add to what you're doing, without replacing anything, would it make sense to explore it?" (wait)


    
    """)
    
    # This method is a tool that the agent can use to get the current weather.
    # The @function_tool decorator exposes this method to the agent's LLM,
    # allowing it to be called when the user asks for the weather.
    # @function_tool()
    # async def get_weather(
    #     self,
    #     context: RunContext,
    #     location: Annotated[
    #         str, "The city and state, e.g. San Francisco, CA"
    #     ],
    # ) -> str:
    #     """Get the current weather in a given location"""
    #     api_key = os.environ.get("OPENWEATHER_API_KEY")
    #     if not api_key:
    #         return "OpenWeather API key is not set."

    #     url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"

    #     async with aiohttp.ClientSession() as session:
    #         async with session.get(url) as response:
    #             if response.status != 200:
    #                 return f"Sorry, I couldn't get the weather. Status code: {response.status}"
                
    #             data = await response.json()
                
    #             if "weather" not in data or not data["weather"]:
    #                 return "Sorry, I couldn't find any weather data for that location."
                
    #             description = data["weather"][0]["description"]
    #             temp = data["main"]["temp"]
                
    #             return f"The weather in {location} is {description} with a temperature of {temp}°C."

# The entrypoint is the main function that runs when a new job for the agent starts.
# It sets up the agent's connection to a LiveKit room and manages its lifecycle.
async def entrypoint(ctx: agents.JobContext):
    # Connect the agent to the LiveKit room associated with the job.
    await ctx.connect()

    # This block attempts to start a recording (egress) of the room's audio.
    # The recording is saved to an S3 bucket.
    try:
        lkapi = api.LiveKitAPI()

        req = api.RoomCompositeEgressRequest(
            room_name=ctx.room.name,
            audio_only=True,
            file_outputs=[
                api.EncodedFileOutput(
                    file_type=api.EncodedFileType.OGG,
                    filepath=f"{ctx.room.name}.ogg",
                    # S3 configuration for uploading the recording.
                    s3=api.S3Upload(
                        access_key=os.environ.get("AWS_S3_ACCESS_KEY"),
                        secret=os.environ.get("AWS_S3_SECRET_KEY"),
                        region="eu-north-1",
                        bucket="livekit-calls"
                    )
                )
            ],
        )
        print("Starting room egress...")
        egress_info = await lkapi.egress.start_room_composite_egress(req)
        await lkapi.aclose()
        egress_id = getattr(egress_info, "egress_id", None) or getattr(egress_info, "egressId", None)
        print(f"Egress started successfully. Egress ID: {egress_id}")
    except Exception as e:
        print(f"Error starting egress: {e}")

    # Check for a phone number in the job metadata to determine if this is an outbound call.
    phone_number = None
    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
            phone_number = metadata.get("phone_number")
        except json.JSONDecodeError:
            print("Error: Invalid JSON in job metadata")

    # If a phone number is provided, initiate an outbound SIP call.
    if phone_number:
        print(f"Attempting to place outbound call to: {phone_number}")
        try:
            # Use the LiveKit API to create a new SIP participant, effectively making a call.
            await ctx.api.sip.create_sip_participant(api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id='ST_S5x7yXbF7QDH', # The specific SIP trunk to use.
                sip_call_to=phone_number,
                participant_identity=phone_number, # Identity for the participant in the room.
                wait_until_answered=True, # Wait for the call to be answered before proceeding.
            ))
            print(f"Call to {phone_number} was answered.")
        except api.TwirpError as e:
            # Handle errors during SIP call creation, like the call not being answered.
            print(f"Error creating SIP participant: {e.message}")
            await ctx.shutdown()
            return

    # Set up the agent's session with various services (plugins).
    session = AgentSession(
        # stt=openai.STT(model="gpt-4o-mini-transcribe"),
        # llm=openai.LLM(model="gpt-4o-mini"),
        # tts=openai.TTS(model="gpt-4o-mini-tts"),
        stt=deepgram.STT(),
        tts=cartesia.TTS(model="sonic-2", voice="73369e4c-fd0c-4f46-92db-01c7fc6ea830"),
        llm=groq.LLM(model="llama3-8b-8192"),
        vad=silero.VAD.load(),
        turn_detection=EnglishModel(),
    )

    # Start the agent session, which begins processing audio from the room.
    await session.start(
        room=ctx.room,
        agent=Assistant(), # Use the Assistant agent we defined earlier.
        room_input_options=RoomInputOptions(
            # Apply noise cancellation to the audio input.
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    # If this is not an outbound call (i.e., no phone number was provided),
    # the agent should start the conversation.
    if not phone_number:
        await session.generate_reply(
            instructions="Greet the user and offer your assistance."
        )

# This is the main execution block. It runs the agent worker when the script is executed
if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="livekit-marek" # A unique name for this agent worker.
    ))