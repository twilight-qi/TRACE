# src/utils/helpers.py
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
from omegaconf import DictConfig

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def send_smtp_email(cfg: DictConfig, subject: str, body: str):
    if not cfg.enabled:
        log.info("Email notification is disabled in the config. Skipping.")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = cfg.smtp.username
    msg["To"] = cfg.recipient_email

    try:
        log.info(f"Connecting to SMTP server {cfg.smtp.host}:{cfg.smtp.port} to send email...")
        if cfg.smtp.use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(cfg.smtp.host, cfg.smtp.port, context=context) as server:
                server.login(cfg.smtp.username, cfg.smtp.password)
                server.send_message(msg)
        else:  # For TLS on port 587
            with smtplib.SMTP(cfg.smtp.host, cfg.smtp.port) as server:
                server.starttls()
                server.login(cfg.smtp.username, cfg.smtp.password)
                server.send_message(msg)

        log.info(f"✅ Email notification sent successfully to {cfg.recipient_email}.")
    except Exception as e:
        log.error(f"❌ Failed to send email notification: {e}")


def send_email_with_dataframe(cfg: DictConfig, subject: str, body: str, metric_data: pd.DataFrame):
    """使用 smtplib 發送郵件通知。"""
    if not cfg.enabled:
        log.info("Email notification is disabled in the config. Skipping.")
        return
    for col in metric_data.select_dtypes(include=["float64", "int64"]).columns:
        metric_data[col] = metric_data[col].apply(lambda x: f"{x:.4g}" if pd.notnull(x) else "")

    html_table = metric_data.to_html(
        index=False,
        escape=False,
        classes="dataframe",
        border=0,
    )

    styled_html = f"""
    <html>
    <head>
    <style>
        table.dataframe {{
            border-collapse: collapse;
            border: 1px solid #ddd;
            width: 100%;
            font-family: Arial, sans-serif;
            font-size: 12px;
        }}
        table.dataframe th {{
            background-color: #f2f2f2;
            color: #333;
            font-weight: bold;
            padding: 8px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        table.dataframe td {{
            padding: 8px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        table.dataframe tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        table.dataframe tr:hover {{
            background-color: #f1f1f1;
        }}
    </style>
    </head>
    <body>
    <p>{body}</p>
        {html_table}
    </body>
    </html>
    """
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = cfg.smtp.username
    msg["To"] = cfg.recipient_email
    msg.attach(MIMEText(styled_html, "html"))
    try:
        log.info(f"Connecting to SMTP server {cfg.smtp.host}:{cfg.smtp.port} to send email...")
        if cfg.smtp.use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(cfg.smtp.host, cfg.smtp.port, context=context) as server:
                server.login(cfg.smtp.username, cfg.smtp.password)
                server.send_message(msg)
        else:  # For TLS on port 587
            with smtplib.SMTP(cfg.smtp.host, cfg.smtp.port) as server:
                server.starttls()
                server.login(cfg.smtp.username, cfg.smtp.password)
                server.send_message(msg)

        log.info(f"✅ Email notification sent successfully to {cfg.recipient_email}.")
    except Exception as e:
        log.error(f"❌ Failed to send email notification: {e}")


def dict_to_dotted_strings(d, parent_key=""):
    result = []
    if d:
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict| DictConfig):
                result.extend(dict_to_dotted_strings(v, new_key))
            else:
                result.append(f"{new_key}={v}")
    else:
        log.warning("Empty dict passed to dict_to_dotted_strings")
    return result