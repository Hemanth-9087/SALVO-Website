import csv
from django.core.management.base import BaseCommand
from django.contrib.auth.hashers import make_password
from django.core.mail import send_mail
from website.models import Member

class Command(BaseCommand):
    help = 'Import members from a CSV file and send credentials email'

    def add_arguments(self, parser):
        parser.add_argument('csv_path', type=str, help='Path to the CSV file')

    def handle(self, *args, **kwargs):
        csv_path = kwargs['csv_path']
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                reg_no = row['register_no']
                raw_password = row['password']
                member, created = Member.objects.get_or_create(
                    register_no=reg_no,
                    defaults={
                        'name': row['name'],
                        'sastra_email': row['sastra_email'],
                        'branch': row['branch'],
                        'batch': row['batch'],
                        'password': make_password(raw_password),
                        'club_role': row['club_role'],
                    }
                )
                if created:
                    email = f"{reg_no}@sastra.ac.in"
                    role = row['club_role']
                    subject = "Welcome to SALVO AI Club - Your Membership Credentials"
                    message = (
                        f"Dear {reg_no},\n\n"
                        f"Congratulations! You have been registered as a {role} in the SALVO AI Club at SASTRA University.\n\n"
                        "As a club member, you are now part of a dynamic network of AI enthusiasts, innovators, and leaders. "
                        "Your role grants you access to exclusive resources, events, and collaborative opportunities to advance your skills and contribute to the club's initiatives.\n\n"
                        "Below are your login credentials for accessing the SALVO AI Club portal:\n\n"
                        f"    Username (Register Number): {reg_no}\n"
                        f"    Password: {raw_password}\n\n"
                        "Please keep this information confidential and secure. For your safety, we do NOT store your password in plain text. "
                        "After logging in for the first time, we strongly recommend that you change your password via your profile page.\n\n"
                        "If you have any questions or need assistance, feel free to reach out to the club coordinators or reply to this email.\n\n"
                        "We are excited to see your contributions and leadership in the SALVO AI Club!\nFor Better Experiance use PC/Desktop Mode.\n\n"
                        "Best regards,\n"
                        "SALVO AI Developer Team\n"
                        "SASTRA University\n"
                        "Email: salvo.aics@gmail.com\n"
                    )
                    send_mail(
                        subject,
                        message,
                        'salvo.aics@gmail.com',
                        [email],
                        fail_silently=False,
                    )
                    self.stdout.write(self.style.SUCCESS(f"Member {reg_no} imported and email sent."))
                else:
                    self.stdout.write(self.style.WARNING(f"Member {reg_no} already exists."))