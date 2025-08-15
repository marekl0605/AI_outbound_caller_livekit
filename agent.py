from __future__ import annotations

import asyncio
import logging
import os
from dotenv import load_dotenv
import json
import time
from typing import Any

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    WorkerOptions,
    RoomInputOptions,
)
from livekit.plugins import (
    deepgram,
    groq,
    elevenlabs,
    silero,
)
from livekit.plugins.turn_detector.english import EnglishModel

# Load environment variables
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.DEBUG)  # Increased logging for latency debugging

# Twilio and LiveKit settings
outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

class OutboundCaller(Agent):
    def __init__(
        self,
        *,
        name: str,
        dial_info: dict[str, Any],
    ):
        super().__init__(
            instructions=f"""
            You are Caleb, a cold caller for Vertex Media (https://www.vertexmedia.us). Engage real estate agents naturally, pitch lead generation services, collect their email, and propose a Zoom appointment. Use filler words ('um,' 'you know,' 'like') to sound human. Ask guiding questions to uncover pain points. Pronounce 'leads' as 'leeds.' Control the call. If asked if you're AI, say you're a Vertex tool and ask a question. Hang up if music is detected or user is unresponsive. Focus on delivering deals.

            **Script**:
            1. **Intro**: 'Hey, is this {name}?' (wait). If name given, say, 'Awesome, just Caleb here. How's it going?' (wait). If asked 'Who?', say, 'Just Caleb from Vertex, first time talking.' Then: 'I know you get tons of calls, but can I have 20 seconds to explain why I'm calling?' (wait, adapt).
            2. **Pitch**: 'Realtors like you face inconsistent months, non-converting leads, or too much work. Which hits you most?' (wait, adapt). Then: 'Vertex works with agents like Coldwell Banker, getting you homeowners ready to sell, handling leads and follow-ups, booking them into your calendar. We use AI funnels, YouTube ads, and ex-agent teams. Could you handle 2-4 extra deals next month?' (wait, handle objections or book). Ask: 'Are you focused on buyers, sellers, listings, or cash flow?' (wait, adapt).
            3. **Booking**: 'We’d love to see if we’re a fit. What’s your time zone?' (wait). Say: 'Does tomorrow work for a Zoom, or the day after?' (wait). Ask: 'Morning or afternoon?' Offer two slots (e.g., 10 AM, 2 PM). Collect: 'What’s your best email? Spell it quick.' (wait 8 seconds). Confirm: 'You’re set for [date/time]. Expect a Vertex email soon—please confirm.' Ask: 'Anything stopping you from attending?' If no, say 'Perfect,' answer questions, wrap up.

            **Objections**:
            - **Not interested/busy**: 'If I could get you 2 deals in 90 days with no work, would you give me a few seconds? You can hang up after.' (wait)
            - **Email/website/text**: 'What are you looking for so I know what to send?' (wait). If pitched: 'Let’s skip email and do a quick call to review testimonials. Sound good?' (wait)
            - **Cost**: 'There’s an investment, but we tailor it after learning your needs. We’ll cover costs on a call, and if we don’t deliver, we work free. Fair?' (wait)
            - **Working with someone**: 'Are you fully satisfied? If we could add value without replacing them, would you explore?' (wait)
            """
        )
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info
        self.email_collected = False
        self.time_zone = None
        self.appointment_date = None
        self.appointment_time = None

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        """Hang up the call by deleting the room"""
        job_ctx = get_job_context()
        try:
            await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
            logger.info(f"Call hung up for {self.participant.identity}")
        except Exception as e:
            logger.error(f"Error hanging up: {e}")

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        """Transfer the call to a human agent after user confirmation"""
        start_time = time.time()
        transfer_to = self.dial_info.get("transfer_to")
        if not transfer_to:
            await ctx.session.generate_reply(instructions="Cannot transfer call, no transfer number provided.")
            logger.debug(f"Transfer call failed: no transfer number, latency: {time.time() - start_time:.3f}s")
            return "cannot transfer call"

        logger.info(f"Transferring call to {transfer_to}")
        await ctx.session.generate_reply(instructions="Let the user know you'll be transferring them.")
        job_ctx = get_job_context()
        try:
            await job_ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job_ctx.room.name,
                    participant_identity=self.participant.identity,
                    transfer_to=f"tel:{transfer_to}",
                )
            )
            logger.info(f"Transferred call to {transfer_to}")
            logger.debug(f"Transfer call completed, latency: {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error transferring call: {e}")
            await ctx.session.generate_reply(instructions="There was an error transferring the call.")
            await self.hangup()
            logger.debug(f"Transfer call failed, latency: {time.time() - start_time:.3f}s")

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """End the call when user requests or conditions met (e.g., music detected)"""
        start_time = time.time()
        logger.info(f"Ending call for {self.participant.identity}")
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()
        await self.hangup()
        logger.debug(f"End call completed, latency: {time.time() - start_time:.3f}s")

    @function_tool()
    async def look_up_availability(self, ctx: RunContext, date: str):
        """Simulate checking appointment availability for demo"""
        start_time = time.time()
        logger.info(f"Looking up availability for {self.participant.identity} on {date}")
        available_times = ["10:00 AM", "2:00 PM"] if "tomorrow" in date.lower() else ["11:00 AM", "3:00 PM"]
        self.appointment_date = date
        logger.debug(f"Availability lookup completed, latency: {time.time() - start_time:.3f}s")
        return {"available_times": available_times}

    @function_tool()
    async def confirm_appointment(self, ctx: RunContext, date: str, time: str, email: str):
        """Simulate confirming appointment for demo"""
        start_time = time.time()
        logger.info(f"Confirming appointment for {self.participant.identity} on {date} at {time}, email: {email}")
        self.email_collected = True
        self.appointment_date = date
        self.appointment_time = time
        logger.debug(f"Appointment confirmation completed, latency: {time.time() - start_time:.3f}s")
        return f"Reservation confirmed for {date} at {time}. You'll receive a confirmation email from Vertex soon—please confirm it."

    @function_tool()
    async def collect_email(self, ctx: RunContext, email: str):
        """Collect and validate user's email"""
        start_time = time.time()
        logger.info(f"Collecting email for {self.participant.identity}: {email}")
        self.email_collected = True
        logger.debug(f"Email collection completed, latency: {time.time() - start_time:.3f}s")
        return f"Email {email} collected, please confirm."

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Hang up if voicemail is detected"""
        start_time = time.time()
        logger.info(f"Detected answering machine for {self.participant.identity}")
        await self.hangup()
        logger.debug(f"Answering machine detection completed, latency: {time.time() - start_time:.3f}s")

async def entrypoint(ctx: JobContext):
    start_time = time.time()
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect()
    logger.debug(f"Room connection completed, latency: {time.time() - start_time:.3f}s")

    dial_info = json.loads(ctx.job.metadata)
    participant_identity = phone_number = dial_info["phone_number"]

    # Initialize agent
    agent = OutboundCaller(
        name="Mustafa",  # Replace with dynamic name from CRM
        dial_info=dial_info,
    )

    # Configure session with ElevenLabs for James-like voice
    session = AgentSession(
        turn_detection=EnglishModel(),  # Kept for turn detection
        vad=silero.VAD.load(),
        stt=deepgram.STT(),  # Removed streaming
        tts=elevenlabs.TTS(voice_id="nXIYu9FT5meibkBbZFT7", model="eleven_multilingual_v2"),  # Removed stream (to test compatibility)
        llm=groq.LLM(model="llama3-8b-8192"),
    )

    # Start session before dialing
    session_start_time = time.time()
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(),
        )
    )
    logger.debug(f"Session start initiated, latency: {time.time() - session_start_time:.3f}s")

    # Dial user with retry logic to handle SIP errors
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            dial_start_time = time.time()
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=outbound_trunk_id,
                    sip_call_to=phone_number,
                    participant_identity=participant_identity,
                    wait_until_answered=True,
                )
            )
            logger.debug(f"SIP participant creation completed, latency: {time.time() - dial_start_time:.3f}s")
            await session_started
            participant = await ctx.wait_for_participant(identity=participant_identity)
            logger.info(f"Participant joined: {participant.identity}")
            agent.set_participant(participant)
            logger.debug(f"Entrypoint completed, total latency: {time.time() - start_time:.3f}s")
            break
        except api.TwirpError as e:
            logger.error(
                f"Attempt {attempt+1} failed creating SIP participant: {e.message}, "
                f"SIP status: {e.metadata.get('sip_status_code')} {e.metadata.get('sip_status')}"
            )
            if attempt == max_attempts - 1:
                ctx.shutdown()
                return
            await asyncio.sleep(1)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="caleb",
        )
    )