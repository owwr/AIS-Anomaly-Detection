import os, sys

import ais
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

"""
Methods for interpreting raw AIS data. This was before we had the
AIS data stream so thankfully this code isn't needed. AIS is not a
fun format to deal with.
"""

def parse_comment_block(block):
    info = {}
    parts = block.split(",")
    for part in parts:
        prefix, data = part.split(":")
        # remove checksums
        if data[-3] == "*":
            data = data[:-3]

        # message group info
        if prefix == "g":
            sentence_num, sentence_tot, group_id = data.split("-")
            info["group"] = {"sentence_num": int(sentence_num), "sentence_tot": int(sentence_tot), "group_id": int(group_id)}
        # message timestamp
        elif prefix == "c":
            info["timestamp"] = data
        # message source
        elif prefix == "s":
            info["source"] = data
        # position quality
        elif prefix == "q":
            info["quality"] = data

    return info

def parse_message_block(block):
    info = {}
    parts = block.split(",")
    if len(parts) != 7:
        print "Error parsing message block " + block
        return

    info["sentence_tot"] = int(parts[1])
    info["sentence_num"] = int(parts[2])
    if parts[3] != "":
        info["sequential_id"] = int(parts[3])
    if parts[4] != "":
        info["channel"] = parts[4]
    info["encoded_message"] = parts[5]
    info["fill_bits"] = int(parts[6][:-3] if parts[6][-3] == "*" else parts[6])

    return info

def parse_information_sentence(sentence):
    parts = sentence[:-1].split("\\")
    if len(parts) != 3:
        print "Error: cound not parse sentence " + sentence

    _, comment_block, message_block = parts
    comment_parsed = parse_comment_block(comment_block)
    message_parsed = parse_message_block(message_block)

    return comment_parsed, message_parsed

def decode_message(lines, first_line_num):
    info = {}
    comment, message = parse_information_sentence(lines[first_line_num])
    if not message:
        return 1, info

    if "timestamp" in comment:
        info["timestamp"] = comment["timestamp"]
    if "source" in comment:
        info["source"] = comment["source"]
    if "quality" in comment:
        info["quality"] = comment["quality"]
    if "channel" in message:
        info["channel"] = message["channel"]
    to_decode = message["encoded_message"]
    fill_bits = message["fill_bits"]

    tot_lines = 1
    if "group" in comment:
        tot_lines = comment["group"]["sentence_tot"]

    for linenum in range(first_line_num + 1, first_line_num + tot_lines):
        _, tempmsg = parse_information_sentence(lines[linenum])
        to_decode += tempmsg["encoded_message"]
        fill_bits = tempmsg["fill_bits"]

    try:
        info["data"] = ais.decode(to_decode, fill_bits)
    except:
        print "Error: could not decode data " + to_decode + " with fill bits " + str(fill_bits)

    return tot_lines, info

def parse_raw_sentences(lines):
    numlines = len(lines)
    currline = 0
    messages = []
    while currline < numlines:
        lines_processed, info = decode_message(lines, currline)
        if "data" in info:
            messages.append(info)
        currline += lines_processed

    return messages

if __name__ == "__main__":
    with open("data/AIS/3.txt", "r") as f:
        print "reading file"
        lines = f.readlines()
        print "done reading file"

        print "parsing lines"
        messages = parse_raw_sentences(lines)
        print "done parsing lines"

        print "grouping by ship mmsi"
        grouped_messages = {}
        for message in messages:
            mmsi = message["data"]["mmsi"]
            grouped_messages[mmsi] = grouped_messages.get(mmsi, [])
            grouped_messages[mmsi].append(message["data"])
        print "done grouping by ship mmsi"

        print len(grouped_messages)

        for mmsi, messages in grouped_messages.items():
            if len(str(mmsi)) != 9:
                continue
            if mmsi != 231211000:
                continue
            print "mmsi: " + str(mmsi)
            xs = []
            ys = []
            for data in messages:
                if "x" in data and "y" in data:
                    xs.append(data["x"])
                    ys.append(data["y"])
            xs = np.array(xs)
            ys = np.array(ys)

            minx = min(xs) - 1*(max(xs) - min(xs))
            maxx = max(xs) + 1*(max(xs) - min(xs))
            miny = min(ys) - 1*(max(ys) - min(ys))
            maxy = max(ys) + 1*(max(ys) - min(ys))
            if miny < -90 or maxy > 90 or minx < -180 or maxx > 180:
                continue

            if minx == maxx or miny == maxy:
                continue

            plt.clf()
            m = Basemap(projection="cyl", llcrnrlat=miny, urcrnrlat=maxy, llcrnrlon=minx, urcrnrlon=maxx, resolution="f")
            m.drawcoastlines()
            m.fillcontinents(color='coral',lake_color='aqua')
            m.drawparallels(np.arange(miny,maxy+0.001,(maxy-miny)/1), labels=[True,True,False,False])
            m.drawmeridians(np.arange(minx,maxx+0.001,(maxx-minx)/1), labels=[False,False,True,True])
            m.drawmapboundary(fill_color='aqua')
            xpts, ypts = m(xs, ys)
            m.scatter(xpts, ypts)
            plt.savefig(os.path.join("ais_info", str(mmsi)+".png"))
