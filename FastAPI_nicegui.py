from fastapi import FastAPI
from nicegui import ui
from nicegui import app as nicegui_app

api = FastAPI()

@api.get('/sum')   # changed here
def api_sum(a: int, b: int):
    return {'sum': a + b}

# Mount FastAPI under /api
nicegui_app.mount('/api', api)

ui.label('FastAPI + NiceGUI Foreground UI')

a = ui.number(label='a', value=5)
b = ui.number(label='b', value=7)
out = ui.label('Result will appear here')

async def compute():
    import httpx
    async with httpx.AsyncClient() as client:
        r = await client.get('http://172.25.86.41:8080/api/sum', params={'a': a.value, 'b': b.value})
        data = r.json()
        if 'sum' in data:
            out.text = f"Sum: {data['sum']}"
        else:
            out.text = f"Error: {data}"
        print(r.status_code, r.text)

ui.button('Compute sum via API', on_click=compute)

ui.run(host='0.0.0.0', port=8080, reload=False)