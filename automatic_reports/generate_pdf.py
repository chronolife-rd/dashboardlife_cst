
import io
import os

from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from PyPDF2 import PdfWriter, PdfReader 

from automatic_reports.config import PATH_PDF
from automatic_reports.config import CstIndicator, CommonIndicator, ImageForPdf

# ------------------------ The main function ---------------------------------
# ----------------------------------------------------------------------------
# def generate_pdf(cst_data_pdf, garmin_data_pdf, common_data_pdf, alerts_dict):
#     in_pdf_path = PATH_PDF + "/empty.pdf"
#     out_pdf_file = PATH_PDF + "/result.pdf"

#     # Delete pdf if exists
#     if os.path.exists(out_pdf_file):
#         os.remove(out_pdf_file)

#     generate_page(cst_data_pdf, garmin_data_pdf, common_data_pdf, alerts_dict, in_pdf_path, out_pdf_file)

# ------------------------ The main function ---------------------------------
# ------------------------------cst only------------------------------------
def generate_pdf(cst_data_pdf,common_data_pdf, alerts_dict):
    in_pdf_path = PATH_PDF + "/empty_cst.pdf"
    out_pdf_file = PATH_PDF + "/result.pdf"

    # Delete pdf if exists
    if os.path.exists(out_pdf_file):
        os.remove(out_pdf_file)

    generate_page(cst_data_pdf,common_data_pdf,alerts_dict, in_pdf_path, out_pdf_file)

# ----------------------- Internal functions ---------------------------------
# ----------------------------cst only--------------------------------------

# def generate_page(cst_data_pdf, garmin_data_pdf, common_data_pdf, alerts_dict, in_pdf_path, out_pdf_file):
def generate_page(cst_data_pdf,common_data_pdf ,alerts_dict, in_pdf_path, out_pdf_file):
    # Costrunct pdf   
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize = A4)

    # GARMIN
    # for garmin_indicator in GarminIndicator:
    #     dict_aux = garmin_data_pdf[garmin_indicator.value]
    #     for key in dict_aux:
    #         _add_text(
    #             can = can, 
    #             text_parameters = dict_aux[key],
    #         )
            
    # CST
    for cst_indicator in CstIndicator:
        dict_aux = cst_data_pdf[cst_indicator.value]
        for key in dict_aux:
            _add_text(
                can = can, 
                text_parameters = dict_aux[key],
            )
    # COMMON
    for common_indicator in CommonIndicator:
        dict_aux = common_data_pdf[common_indicator.value]
        for key in dict_aux:
            _add_text(
                can = can, 
                text_parameters = dict_aux[key],
            )
    
    # ALERTS
    for key in alerts_dict:
        _add_image(
            can = can, 
            image_parameters = alerts_dict[key],
        )
    
    # IMAGES
    for image_parameters in ImageForPdf:
        _add_image(can, image_parameters.value)
        
    can.showPage()
    can.save()

    # Move to the beginning of the StringIO buffer
    packet.seek(0)

    # Create a new PDF with Reportlab
    new_pdf = PdfReader(packet)
    
    # Read your existing PDF
    existing_pdf = PdfReader(open(in_pdf_path, "rb"))
    output = PdfWriter()

    # Add the "watermark" (which is the new pdf) on the existing page
    # reader.pages[page_number]
    page = existing_pdf.pages[0]
    page.merge_page (new_pdf.pages[0])
    output.add_page(page)

    # Finally, write "output" to a real file
    output_stream = open(out_pdf_file, "wb")
    output.write(output_stream)
    output_stream.close()

def _add_text(can, text_parameters):
    text = text_parameters["text"]
    x = text_parameters["x"]
    y = text_parameters["y"]
    color = text_parameters["color"]
    font = text_parameters["font"]
    size = text_parameters["size"]

    if isinstance(x, str) == False:
        y_top = 11.69
        x_start = x*inch
        y_start = (y_top - y)*inch
        
        can.setFillColor(color)
        can.setFont(font, size)
        can.drawString(x_start, y_start, text)

def _add_image(can, image_parameters):
    x = image_parameters["x"]
    y = image_parameters["y"]
    width = image_parameters["w"]
    height = image_parameters["h"]
    path_img = image_parameters["path"]

    if isinstance(x, str) == False and os.path.exists(path_img):
        width = width*inch 
        height = height*inch

        y_top = 11.69
        x_start = x*inch
        y_start = (y_top - y)*inch 

        can.drawImage(path_img, x_start, y_start, width, height,
                    preserveAspectRatio = True, mask = 'auto')