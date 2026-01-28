// bridge.js
import express from 'express';
import { WebSocketServer } from 'ws';
import cors from 'cors';
import bodyParser from 'body-parser';

const app = express();
const HTTP_PORT = 3001;
const WS_PORT = 3002;

const wss = new WebSocketServer({ port: WS_PORT });

wss.on('connection', (ws) => {
    console.log('Frontend connected to bridge');
});

function broadcast(data) {
    wss.clients.forEach(client => {
        if (client.readyState === 1) client.send(data);
    });
}

app.use(cors());

app.use(bodyParser.raw({ type: 'image/png', limit: '10mb' }));

app.post('/upload', (req, res) => {
    console.log(`Received image from Figma: ${req.body.length} bytes`);
    
    broadcast(req.body);
    
    res.status(200).send('OK');
});

app.listen(HTTP_PORT, () => {
    console.log(`Bridge listening: HTTP ${HTTP_PORT} -> WS ${WS_PORT}`);
});