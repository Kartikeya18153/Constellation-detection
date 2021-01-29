# Constellation-detection
Image processing tool and algorithm for star constellation detection from the image. 
<p align="center">
  <img src="https://media.giphy.com/media/3orieRv4TCLD3NduoM/giphy.gif" />
</p>





We had used list of constellations published by International
Astronomical Union (IAU), to maintain a standard convention. We have used this th

A constellation is a recognized pattern of stars in the night sky. There are a total of 88 known constellations.
During the ancient times, certain constellations had acquired special significance over time because of their
appearance marking the start of a new season, guiding travelers and people,
letting farmers know when to sow or reap a crop, and hunters to tell the best time to hunt. They also played a part
in being the first GPS, and still play an important role in
satellite placement and positioning. A list of constellations has now been published by International
Astronomical Union (IAU). While identifying constellations is not very difficult for people who are experience in the
activity of stargazing, it can be difficult for newcomers to explore the numerous patterns and the corresponding stars
involved in the making of those patters.
We try to apply our knowledge of various image processing techniques and algorithms to help in the
detection of these constellations in the night sky. Our main focus is to build a detection tool for naked eye

constellations. The main detection tool are the templates we
use, against which a test is compared and try to figure out
the constellation in that test image. Along with processing
the test images, the templates had to be processed as well to
make them usable.
There were three important steps in our constellation
detection algorithm. The first was the creation of the
template database. The original templates were obtained
from an image gallery of constellations [3]. We select the 30
largest and most prominent constellations present in the
night sky using a list available online [4]. Various image
processing techniques are the applied to create the template
database which is used in detection algorithm. The template
creation is implemented based on an existing set of modified
constellation charts [5]. The second was the processing of
test images. The test images were obtained from a night sky
observation application [6] and then processed to make
detection of constellations feasible. The third was creation
of a detection algorithm, that would let us detect a
constellation irrespective of the way it was present in the
image. The performed procedures and techniques are
discussed in detail in the following sections.

-----

# Table of contents


- [Project Title](#analyzing-sentiments-from-social-media)
- [Demo-Preview](#demo-preview)
- [Table of contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)
- [License](#license)
- [Footer](#footer)

# Installation

To use this project, first clone this git repository using following commands

    git init
    git clone https://github.com/Kartikeya18153/Constellation-detection.git

Then comes the, installation of dependences

I have use following dependences
- numpy
- matplotlib
- cv2
- math
- os
- pickle
- copy

for installing all the dependencies, use requirement file with following command

    pip3 install -r requirment.txt
<!-- # TODO : need to define requiremtn.txt -->
# Usage

You need to run the [DIP_Project.py](https://github.com/Kartikeya18153/Constellation-detection/blob/master/DIP_Project.py). It will crunch all the templates and test data to form Normalised immages for genrating results.

# Authors

- [AnikaitSahota](https://github.com/AnikaitSahota)

- [Kartikeya Gupta](https://github.com/Kartikeya18153)

See also the list of [contributors](https://github.com/Kartikeya18153/Constellation-detection/contributors) who participated in this project.

# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

# Footer
- If you want to contribute, fork the repository for yourself. Check out [Fork list](https://github.com/Kartikeya18153/Constellation-detection/network/members)
- If you liked the project, leave a star. Check out [stargazers](https://github.com/Kartikeya18153/Constellation-detection/stargazers)




TODO : 
  * delete the unrequired file
    * All images in main directory (like final.png, test.png)
    * Report doc file
