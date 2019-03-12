from .qub_model_back import consumers

channel_routing = {
    'websocket.connect': consumers.ws_connect,
    'websocket.receive': consumers.ws_receive,
    'websocket.disconnect': consumers.ws_disconnect,
}