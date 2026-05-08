"""
Escalation Engine for Pothole Detection System
================================================
Automatically detects overdue pothole complaints and sends a single
escalation email to the Senior Engineer when the repair deadline is missed.

Time limits by severity:
  - High:   5 days
  - Medium: 8 days
  - Low:    15 days

Escalation: one email goes to the Senior Engineer as soon as the deadline passes.
"""

import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

# SMTP Configuration (Gmail)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "higherofficialroad@gmail.com"
SENDER_PASSWORD = "ijgi ecqw edmb aefw"  # Gmail App Password

# Severity-based repair deadlines (in days)
SEVERITY_DEADLINES = {
    "High": 5,
    "Medium": 8,
    "Low": 15,
}

# Single escalation target — Senior Engineer is notified when any deadline is missed
SENIOR_ENGINEER = {
    "title": "Senior Engineer",
    "email": "higherofficialroad@gmail.com",
}

# File paths
SUBMISSIONS_FILE = "submissions.json"
NOTIFICATIONS_FILE = "notifications.json"
ESCALATION_EMAILS_FILE = "escalation_emails.json"


# ============================================================================
# DATA ACCESS
# ============================================================================

def load_submissions():
    """Load all submissions from JSON file."""
    if not os.path.exists(SUBMISSIONS_FILE):
        return []
    with open(SUBMISSIONS_FILE, "r") as f:
        return json.load(f)


def save_submissions(submissions):
    """Save all submissions back to JSON file."""
    with open(SUBMISSIONS_FILE, "w") as f:
        json.dump(submissions, f, indent=4)


def load_notifications():
    """Load notifications from JSON file."""
    if not os.path.exists(NOTIFICATIONS_FILE):
        return []
    with open(NOTIFICATIONS_FILE, "r") as f:
        return json.load(f)


def save_notifications(notifications):
    """Save notifications to JSON file."""
    with open(NOTIFICATIONS_FILE, "w") as f:
        json.dump(notifications, f, indent=4)


def load_escalation_emails():
    """Load escalation email log."""
    if not os.path.exists(ESCALATION_EMAILS_FILE):
        return []
    with open(ESCALATION_EMAILS_FILE, "r") as f:
        return json.load(f)


def save_escalation_emails(emails):
    """Save escalation email log."""
    with open(ESCALATION_EMAILS_FILE, "w") as f:
        json.dump(emails, f, indent=4)


# ============================================================================
# DEADLINE CALCULATION
# ============================================================================

def calculate_deadline(submission_timestamp, severity):
    """
    Calculate the repair deadline based on when the report was submitted
    and its severity level.
    
    Args:
        submission_timestamp: String in format "YYYY-MM-DD HH:MM:SS"
        severity: "High", "Medium", or "Low"
    
    Returns:
        Deadline as a string in format "YYYY-MM-DD"
    """
    if severity not in SEVERITY_DEADLINES:
        return None
    
    submitted_date = datetime.strptime(submission_timestamp, "%Y-%m-%d %H:%M:%S")
    deadline_days = SEVERITY_DEADLINES[severity]
    deadline_date = submitted_date + timedelta(days=deadline_days)
    return deadline_date.strftime("%Y-%m-%d")


def get_days_overdue(deadline_str):
    """
    Calculate how many days past the deadline we are.
    
    Returns:
        Number of days overdue (positive = overdue, negative/zero = not yet due)
    """
    if not deadline_str:
        return 0
    deadline = datetime.strptime(deadline_str, "%Y-%m-%d")
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return (today - deadline).days


# ============================================================================
# EMAIL COMPOSITION
# ============================================================================

def compose_escalation_email(submission, days_overdue):
    """
    Compose a formal escalation email with full report details.
    Sent to the Senior Engineer when a repair deadline is missed.
    
    Returns:
        (subject, body_html) tuple
    """
    report_id = submission.get("id", "N/A")
    severity = submission.get("severity", "Unknown")
    severity_score = submission.get("severity_score", 0)
    latitude = submission.get("latitude", "N/A")
    longitude = submission.get("longitude", "N/A")
    timestamp = submission.get("timestamp", "N/A")
    status = submission.get("status", "N/A")
    deadline = submission.get("deadline", "N/A")

    # Location string
    if latitude and longitude and latitude != "N/A":
        location_str = f"{latitude}, {longitude}"
        map_link = f"https://www.google.com/maps?q={latitude},{longitude}"
    else:
        location_str = "GPS coordinates not available"
        map_link = None

    target_title = SENIOR_ENGINEER["title"]

    subject = f"🚨 URGENT: Pothole Repair Overdue — Report ID: {report_id} | {severity} Severity | {days_overdue} Days Overdue"

    body_html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 650px; margin: 0 auto; border: 2px solid #dc3545; border-radius: 10px; overflow: hidden;">
            
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #dc3545, #c82333); color: white; padding: 20px 25px;">
                <h1 style="margin: 0; font-size: 22px;">🚨 POTHOLE REPAIR ESCALATION NOTICE</h1>
                <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;">Automated Alert from CivicAI Pothole Detection System</p>
            </div>
            
            <!-- Body -->
            <div style="padding: 25px;">
                <p>Dear <strong>{target_title}</strong>,</p>
                
                <p>This is to bring to your <strong>urgent attention</strong> that the following pothole complaint has 
                <span style="color: #dc3545; font-weight: bold;">NOT been repaired within the stipulated time limit</span> 
                and requires your immediate intervention.</p>
                
                <!-- Report Details Box -->
                <div style="background: #f8f9fa; border-left: 4px solid #dc3545; padding: 15px 20px; margin: 20px 0; border-radius: 0 8px 8px 0;">
                    <h3 style="margin: 0 0 12px 0; color: #dc3545;">📋 Complaint Details</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr><td style="padding: 6px 0; font-weight: bold; width: 180px;">Report ID:</td><td>{report_id}</td></tr>
                        <tr><td style="padding: 6px 0; font-weight: bold;">Reported On:</td><td>{timestamp}</td></tr>
                        <tr><td style="padding: 6px 0; font-weight: bold;">Severity:</td><td><span style="background: {'#dc3545' if severity == 'High' else '#fd7e14' if severity == 'Medium' else '#28a745'}; color: white; padding: 2px 10px; border-radius: 4px;">{severity}</span> (Score: {severity_score:.1%})</td></tr>
                        <tr><td style="padding: 6px 0; font-weight: bold;">Location (GPS):</td><td>{location_str}</td></tr>
                        {f'<tr><td style="padding: 6px 0; font-weight: bold;">Map Link:</td><td><a href="{map_link}" style="color: #007bff;">View on Google Maps</a></td></tr>' if map_link else ''}
                        <tr><td style="padding: 6px 0; font-weight: bold;">Current Status:</td><td>{status}</td></tr>
                    </table>
                </div>
                
                <!-- Deadline Info Box -->
                <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px 20px; margin: 20px 0; border-radius: 0 8px 8px 0;">
                    <h3 style="margin: 0 0 12px 0; color: #856404;">⏰ Deadline Information</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr><td style="padding: 6px 0; font-weight: bold; width: 180px;">Repair Deadline:</td><td><strong>{deadline}</strong></td></tr>
                        <tr><td style="padding: 6px 0; font-weight: bold;">Days Overdue:</td><td><span style="color: #dc3545; font-weight: bold; font-size: 18px;">{days_overdue} DAYS</span></td></tr>
                        <tr><td style="padding: 6px 0; font-weight: bold;">Escalated To:</td><td><strong>{target_title}</strong></td></tr>
                    </table>
                </div>
                
                <!-- Action Required -->
                <div style="background: #d4edda; border-left: 4px solid #28a745; padding: 15px 20px; margin: 20px 0; border-radius: 0 8px 8px 0;">
                    <h3 style="margin: 0 0 8px 0; color: #155724;">✅ Action Required</h3>
                    <p style="margin: 0;">As the {target_title}, you are requested to:</p>
                    <ol style="margin: 8px 0;">
                        <li>Take <strong>immediate cognizance</strong> of this complaint</li>
                        <li>Ensure the pothole at the reported location is <strong>inspected and repaired on priority</strong></li>
                        <li>Update the complaint status on the CivicAI system once work is completed</li>
                        <li>Upload a verification photo of the repaired road</li>
                    </ol>
                </div>
                
                <p style="margin-top: 25px; padding-top: 15px; border-top: 1px solid #dee2e6; font-size: 13px; color: #666;">
                    This is an automated escalation email generated by the <strong>CivicAI Pothole Detection & Severity Analysis System</strong>. 
                    The complaint was automatically escalated because the repair was not completed within the 
                    <strong>{SEVERITY_DEADLINES.get(severity, 'N/A')}-day</strong> time limit mandated for <strong>{severity}</strong>-severity potholes.
                </p>
                
                <p style="font-size: 13px; color: #666;">
                    Failure to address this complaint may result in further escalation to higher authorities.
                </p>
                
                <p style="margin-top: 20px;">
                    Regards,<br>
                    <strong>CivicAI Automated Escalation System</strong><br>
                    Pothole Detection & Severity Analysis Platform
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return subject, body_html


# ============================================================================
# EMAIL SENDING
# ============================================================================

def send_email(to_email, subject, body_html):
    """
    Send an email via Gmail SMTP.
    
    Returns:
        (success: bool, message: str)
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = f"CivicAI Escalation System <{SENDER_EMAIL}>"
        msg["To"] = to_email
        msg["Subject"] = subject
        
        # Attach HTML body
        msg.attach(MIMEText(body_html, "html"))
        
        # Connect to Gmail SMTP and send
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        server.quit()
        
        return True, "Email sent successfully"
    
    except smtplib.SMTPAuthenticationError:
        return False, "SMTP Authentication failed — check email/app password"
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {str(e)}"
    except Exception as e:
        return False, f"Error sending email: {str(e)}"


# ============================================================================
# ESCALATION LOGIC
# ============================================================================


def run_escalation_check():
    """
    Main escalation function. Scans all submissions for overdue reports
    and sends a single escalation email to the Senior Engineer when the
    repair deadline is first missed. Will not re-send if already escalated.
    
    Returns:
        List of escalation actions taken (for display in the dashboard)
    """
    submissions = load_submissions()
    notifications = load_notifications()
    email_log = load_escalation_emails()
    
    actions_taken = []
    updated = False
    
    for sub in submissions:
        # Skip resolved or already-escalated complaints
        if sub.get("status") in ["Resolved", "Escalated"]:
            continue
        
        severity = sub.get("severity")
        if not severity:
            continue  # Not yet analyzed
        
        # Ensure deadline is set
        if not sub.get("deadline"):
            deadline = calculate_deadline(sub["timestamp"], severity)
            if deadline:
                sub["deadline"] = deadline
                updated = True
        
        deadline = sub.get("deadline")
        if not deadline:
            continue
        
        days_overdue = get_days_overdue(deadline)
        if days_overdue <= 0:
            continue  # Not yet overdue
        
        # Mark as overdue
        sub["status"] = "Overdue"
        updated = True

        # Send ONE escalation email to the Senior Engineer
        target_email = SENIOR_ENGINEER["email"]
        target_title = SENIOR_ENGINEER["title"]
        
        subject, body_html = compose_escalation_email(sub, days_overdue)
        email_sent, email_message = send_email(target_email, subject, body_html)
        
        # Log the escalation email
        email_record = {
            "id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(100000, 999999)}",
            "complaint_id": sub["id"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "to_authority": target_title,
            "to_email": target_email,
            "subject": subject,
            "days_overdue": days_overdue,
            "severity": severity,
            "email_sent": email_sent,
            "email_status": email_message,
            "deadline": deadline,
        }
        email_log.append(email_record)
        
        # Mark complaint as Escalated so it won't be re-sent next session
        sub["status"] = "Escalated"
        sub["assigned_to"] = target_title
        updated = True
        
        # Add notification
        notifications.append({
            "id": email_record["id"],
            "complaint_id": sub["id"],
            "authority": target_title,
            "message": f"📧 Email sent to {target_title} ({target_email}) — Complaint {sub['id']} is {days_overdue} days overdue",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "read": False,
            "email_sent": email_sent,
        })
        
        actions_taken.append({
            "complaint_id": sub["id"],
            "severity": severity,
            "days_overdue": days_overdue,
            "escalated_to": target_title,
            "to_email": target_email,
            "email_sent": email_sent,
            "email_status": email_message,
            "deadline": deadline,
        })
    
    # Save everything
    if updated:
        save_submissions(submissions)
    save_notifications(notifications)
    save_escalation_emails(email_log)
    
    return actions_taken


def get_overdue_summary():
    """
    Get a summary of all overdue complaints and their escalation status.
    Used for dashboard display.
    
    Returns:
        List of dicts with overdue complaint info
    """
    submissions = load_submissions()
    overdue = []
    
    for sub in submissions:
        if sub.get("status") == "Resolved":
            continue
        
        severity = sub.get("severity")
        deadline = sub.get("deadline")
        
        if not severity or not deadline:
            continue
        
        days_overdue = get_days_overdue(deadline)
        if days_overdue > 0:
            overdue.append({
                "id": sub["id"],
                "severity": severity,
                "deadline": deadline,
                "days_overdue": days_overdue,
                "status": sub.get("status", "Unknown"),
                "assigned_to": sub.get("assigned_to", "Unknown"),
                "escalation_level": sub.get("escalation_level", 0),
                "timestamp": sub.get("timestamp", ""),
                "latitude": sub.get("latitude"),
                "longitude": sub.get("longitude"),
            })
    
    # Sort by days overdue (most urgent first)
    overdue.sort(key=lambda x: x["days_overdue"], reverse=True)
    return overdue


def get_email_log():
    """Get the escalation email log for dashboard display."""
    return load_escalation_emails()


# ============================================================================
# STANDALONE EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚨 Running Escalation Check...")
    print("=" * 60)
    
    actions = run_escalation_check()
    
    if actions:
        print(f"\n📧 {len(actions)} escalation(s) processed:\n")
        for a in actions:
            status_icon = "✅" if a["email_sent"] else "❌"
            print(f"  {status_icon} Complaint {a['complaint_id']}")
            print(f"     Severity: {a['severity']} | {a['days_overdue']} days overdue")
            print(f"     Escalated to: {a['escalated_to']} ({a['to_email']})")
            print(f"     Email: {a['email_status']}")
            print()
    else:
        print("\n✅ No new escalations needed.")
    
    overdue = get_overdue_summary()
    if overdue:
        print(f"\n⚠️  {len(overdue)} overdue complaint(s):")
        for o in overdue:
            print(f"  - {o['id']}: {o['severity']} severity, {o['days_overdue']} days overdue")
