from channels import Group
from channels.sessions import channel_session


@channel_session
def ws_connect(message):
    print('connected')
    return

@channel_session
def ws_recieve(message):
    print('received')
    return

@channel_session
def ws_receive(message):
    print('received')
    return


@channel_session
def ws_disconnect(message):
    print("disconnected")
    return

