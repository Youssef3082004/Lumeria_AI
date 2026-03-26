# استخدام نسخة بايثون خفيفة
FROM python:3.9

# تحديد مكان العمل داخل السيرفر
WORKDIR /code

# نسخ ملف المكتبات وتثبيتها
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# نسخ باقي ملفات الكود
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]