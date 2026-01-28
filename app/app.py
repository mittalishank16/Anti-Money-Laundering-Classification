import gradio as gr
import requests
import pandas as pd

def call_api(from_bank, to_bank, payment_format, receiving_currency, payment_currency, amount_received, amount_paid):
    payload = {
        "from_bank": from_bank,
        "to_bank": to_bank,
        "payment_format": payment_format,
        "receiving_currency": receiving_currency,
        "payment_currency": payment_currency,
        "amount_received": amount_received,
        "amount_paid": amount_paid
    }
    
    # Pointing to the FastAPI container address
    response = requests.post("http://localhost:8000/predict", json=payload)
    result = response.json()
    return f"Risk Level: {result['status']}", f"Fraud Probability: {result['probability']}"

# Define UI
with gr.Blocks(title="AML Transaction Monitor") as demo:
    gr.Markdown("# 🏦 Anti-Money Laundering Detection")
    
    with gr.Row():
        from_b = gr.Number(label="From Bank ID", value=10)
        to_b = gr.Number(label="To Bank ID", value=20)
    
    with gr.Row():
        pay_fmt = gr.Dropdown(['Cheque', 'Credit Card', 'ACH', 'Others'], label="Payment Format")
        rec_cur = gr.Dropdown(['US Dollar', 'Euro', 'Swiss Franc', 'Yuan', 'Others'], label="Receiving Currency")
        pay_cur = gr.Dropdown(['US Dollar', 'Euro', 'Swiss Franc', 'Yuan', 'Others'], label="Payment Currency")
        
    with gr.Row():
        amt_rec = gr.Number(label="Amount Received")
        amt_paid = gr.Number(label="Amount Paid")
        
    btn = gr.Button("Analyze Transaction", variant="primary")
    
    with gr.Row():
        out_status = gr.Textbox(label="Result")
        out_prob = gr.Textbox(label="Confidence Score")

    btn.click(call_api, inputs=[from_b, to_b, pay_fmt, rec_cur, pay_cur, amt_rec, amt_paid], outputs=[out_status, out_prob])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)