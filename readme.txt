INFO:     Finished server process [301283]                                                                                      nohup uvicorn main:app --host 127.0.0.1 --port 8000 > ai.log 2>&1 &
[1] 301309t@srv880832:/var/www/OxfordOnline/AI# nohup uvicorn main:app --host 127.0.0.1 --port 8000 > ai.log 2>&1 &
(venv) root@srv880832:/var/www/OxfordOnline/AI# 
(venv) root@srv880832:/var/www/OxfordOnline/AI# ps aux | grep uvicorn
root      301309 41.7  9.8 4929556 801432 ttyS0  Sl   14:11   0:06 /var/www/OxfordOnline/AI/venv/bin/python3 /var/www/OxfordOnline/AI/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
root      301318  0.0  0.0   7076  2292 ttyS0    S+   14:11   0:00 grep --color=auto uvicorn
(venv) root@srv880832:/var/www/OxfordOnline/AI# 

Swagger:

https://oxfordonline.com.br/AI/docs

OpenAPI:

https://oxfordonline.com.br/AI/openapi.json

Para ver logs da IA futuramente:

tail -f ai.log

Para parar a API:

kill 301309

ou:

pkill -f uvicorn

Reestartar nginx:  sudo systemctl restart nginx

------------------------------------------------ ULTIMAS MELHORIAS ---------------------------------------------------------

pip install scikit-image opencv-python transformers timm torch torchvision

source venv/bin/activate
nohup uvicorn main:app --host 127.0.0.1 --port 8000 > ai.log 2>&1 &

uvicorn main:app --reload

------------------------------------------------ PARA REESTARTAR ------------------------------------------------

Ignorar as pastas: venv/, __pycache__/, .git/.

pip install -r requirements.txt

//---------------------------------------- NO PC LOCAL ----------------------------------------

uvicorn main:app --reload --host 0.0.0.0 --port 8000

//---------------------------------------- API PYTHON ----------------------------------------

MATA
sudo lsof -i :8000
sudo kill -9 324069

INICIA UNICIORN:
./venv/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &

INICIA:
uvicorn main:app --host 0.0.0.0 --port 8000 &

sudo systemctl reload nginx

//---------------------------------------- SERVIÇO API PYTHON ----------------------------------------
sudo nano /etc/systemd/system/oxf-ai.service

Ativar e Iniciar o Novo Serviço
Agora que o serviço foi criado, vamos derrubar aquele processo manual que você iniciou e passar o controle para o sistema:

Bash
# 1. Recarrega o sistema para reconhecer o novo serviço
sudo systemctl daemon-reload

# 2. Ativa o serviço para iniciar automaticamente caso o servidor seja reiniciado
sudo systemctl enable oxf-ai.service

# 3. Para o processo antigo que estava na porta 8000 para não dar conflito
sudo kill -9 340656

# 4. Inicia a API através do novo serviço definitivo
sudo systemctl start oxf-ai.service


sudo systemctl daemon-reload
sudo systemctl restart oxf-ai.service
sudo systemctl status oxf-ai.service

para logs
sudo journalctl -u oxf-ai.service -f

//--------------------------------------------------------------------------------------------