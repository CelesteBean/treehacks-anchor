"""Message bus monitor - run on Jetson display."""
import zmq
from src.core.message_bus import MessageBus, AUDIO_PORT, TRANSCRIPT_PORT

bus = MessageBus()
sub = bus.create_subscriber([AUDIO_PORT, TRANSCRIPT_PORT])

print('=== Message Bus Monitor ===')
print('Listening on ports 5555 (audio), 5556 (transcript)')
print('Ctrl+C to exit\n')

audio_count = 0
while True:
    result = bus.receive(sub, timeout_ms=100)
    if result:
        topic, data = result
        if topic == 'audio':
            audio_count += 1
            if audio_count % 10 == 0:
                print(f'[AUDIO] chunks received: {audio_count}')
        else:
            print(f'[{topic.upper()}] {data}')
