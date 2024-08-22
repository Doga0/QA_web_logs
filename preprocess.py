#import libraries
import json
import csv

# load HAR file
with open("www.tr.freelancer.com.har", 'r', encoding="utf-8") as har_file:
    data = json.load(har_file)

# data to extract
labels = ["IP", "method", "url", "date", "location", "page_id", "page_title", "status", 
          "status_text", "response_size", "request_time", "send","wait", "receive"]

# create and open CSV file
with open("data.csv", "a", encoding="utf-8") as csv_file:
    # Write the header row to the CSV file
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(labels)
    
    # extract entries and pages
    entries = data["log"]["entries"]
    pages = data["log"]["pages"]

    # extract page information
    for page in pages:
        page_id = page["id"]
        page_title = page["title"]

    # loop through each entry
    for entry in entries:
        # extract IP adress
        ip_address = entry["serverIPAddress"] 
        
        # extract url and method
        request = entry["request"]
        url = request["url"]
        method = request["method"]
        
        # extract status, status_text and response_size
        response = entry["response"]
        headers = response["headers"]
        status = response["status"]
        status_text = response["statusText"]
        content = response["content"]
        response_size = content["size"]
        
        # extract request_time
        request_time = entry["time"]
        
        # extract timings
        timings = entry["timings"]
        send = timings["send"]
        wait = timings["wait"]
        receive = timings["receive"]

        date = ""
        location = ""

        # loop through headers
        for header in headers:
            # extract date
            if header["name"] == 'date':
                date = header["value"]
            
            #extract location
            if header["name"] == "x-edge-location":
                location = header["value"].split("-")[0]
        
        # combine all extracted data
        row = [
            ip_address, method, url, date, location,
            page_id, page_title, status,
            status_text, response_size, request_time, send,
            wait, receive
        ]
        print(response_size, url)
        #print(row)

        # append data to CSV file
        csv_writer.writerow(row)
