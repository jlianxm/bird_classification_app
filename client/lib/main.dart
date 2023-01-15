// ignore_for_file: unused_import, use_key_in_widget_constructors, unused_local_variable, prefer_const_literals_to_create_immutables, prefer_const_constructors, deprecated_member_use, sized_box_for_whitespace, avoid_print
// https://www.porkaone.com/2022/07/how-to-upload-images-and-display-them.html

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:async/async.dart';
import 'package:path/path.dart' as path;
import 'dart:convert';

void main() => runApp(MaterialApp(
  home: Home(),
  debugShowCheckedModeBanner: false,
));

class Home extends StatefulWidget {
  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {

  XFile? image;

  final ImagePicker picker = ImagePicker();

  //we can upload image from camera or from gallery based on parameter
  Future getImage(ImageSource media) async {
    var img = await picker.pickImage(source: media);

    setState(() {
      image = img;
    });
  }

  //show popup dialog
  void myAlert() {
    showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog (
            shape:
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
            title: Text('Please choose media to select'),
            content: Container(
              height: MediaQuery.of(context).size.height / 6,
              child: Column(
                children: [
                  ElevatedButton(
                    //if user click this button, user can upload image from gallery
                    onPressed: () {
                      Navigator.pop(context);
                      getImage(ImageSource.gallery);
                    },
                    child: Row(
                      children: [
                        Icon(Icons.image),
                        Text('From Gallery'),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          );
        });
  }

  Future<Prediction> uploadImageToServer(File imageFile) async {
    var stream =
    http.ByteStream(DelegatingStream(imageFile.openRead()));
    var length = await imageFile.length();

    var uri = Uri.parse('http://172.18.201.250:8080/up_photo');
    var request = http.MultipartRequest("POST", uri);
    // var headers = {"Content-type": "multipath/form-data"};

    var multipartFile = http.MultipartFile('file', stream, length,
        filename: path.basename(imageFile.path));
    request.files.add(multipartFile);
    // request.headers.addAll(headers);
    print("Request: " + request.toString());

    var streamedResponse = await request.send();
    var response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      print(jsonDecode(response.body));
      return Prediction.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Error');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Upload Image'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () {
                myAlert();
                // Navigator.pop(context);
                // getImage(ImageSource.gallery);
              },
              child: Text('Select Photo'),
            ),
            SizedBox(
              height: 10,
            ),
            ElevatedButton(
              onPressed: () {
                if (image != null) {
                  uploadImageToServer(File(image!.path));
                }
              },
              child: Text('Upload Photo'),
            ),
            SizedBox(
              height: 10,
            ),
            //if image not null show the image
            //if image null show text
            image != null
                ? Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.file(
                  //to show image, you type like this.
                  File(image!.path),
                  fit: BoxFit.cover,
                  width: MediaQuery.of(context).size.width,
                  height: 300,
                ),
              ),
            )
                : Text(
              "No Image",
              style: TextStyle(fontSize: 20),
            )
          ],
        ),
      ),
    );
  }
}

class Prediction {
  final String predictedClass;

  Prediction({required this.predictedClass});

  factory Prediction.fromJson(Map<String, dynamic> json) {
    return Prediction(
      predictedClass: json['result'],
    );
  }
}

