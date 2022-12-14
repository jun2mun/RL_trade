@Echo off

cd C:\Users\owner\Desktop\junbeom\RL_trade
call C:\Users\owner\anaconda3\Scripts\activate.bat C:\Users\owner\anaconda3
call conda activate RLtrader

call python -m streamlit run app.py

echo hi

pause