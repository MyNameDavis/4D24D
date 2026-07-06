# **4D24D Overview**

The purpose of the 4D24D toolset is to allow users to easily get high-res scans of their film
using a DSLR (Digital Single Lens Reflector) even if it has a low resolution sensor. This is achieved
by taking multiple photographs of partial segments of the image, and then combining them into one
"super-resolution" mosaic image. 4D24D makes this easier by automatically detecting groups of images
which belong to the same film photo, correcting for slight camera misalignments, and lighting inconistencies,
and stiching the image together into a perfectly aligned and cropped output without any user input.
You can take as many photos as you want, of film of any aspect ratio, and put them in any order,
and 4D24D will figure the rest out for you!

The purpose of this project is to make high resolution film photography scans available at low
cost to amateur photographers. Anybody who's ever shot on film knows that it's not cheap!
Getting high resolution scans is no small part of that cost. At home scanning alternatives
exist but are also very expensive. Even if you ditch a dedicated film scanner in favor of the more
accessible DSLR film scanning method, stands to attach your DSLR to and gaurantee perfect alignment
are incredibly pricy. Not to mention the cost of a nice, high resolution DSLR and macro lens. I built
4D24D so that even photographers without an expensive DLSR, macro lens, scanner or whatever else
can still get quality scans of the film photos they've shot.

## **Minimal Hardware Requirements to Get Good Results with 4D24D**

As previously mentioned, 4D24D is purpose built to get you great scans with a cheap DSLR, and no
macro lens. While highly recommended, you don't even strictly need a stand or tripod to mount your
DSLR to. If you've got steady enough hands, 4D24D can auto correct for any slight misalignments between
the film and your camera sensor (you'll need VERY steady hands). With that said, here's what I recommend
you have at a minimum before using 4D24D:

* Any DSLR + lens
* Camera lens reversing ring - this lets you mount your lens to your camera backwards to simulate the
    effects of a macro lens. It won't be quite as good, but it'll let you take in-focus shots of your
    film from very close up, which is a must to get the benefits of 4D24D.
* Backlight for film

[A note for stanky little degenerates: you can *technically* use any camera and lens for taking
partial segment photos. I've even tested it out with an iPhone. Just don't expect immaculate outcomes.]

### Optional but highly recommended for best results:
* A tripod/camera stand and mount. Helps you take consistent, level shots. The higher your target resolution,
  the more necessary this becomes.
* A film holder. Keeps your film flat, and allows you to slide shots across the backlight more easily.

### Above and beyond items for those seeking the chef's kiss:##
* A focusing rail to dial in your camera's alignment and focus with precision.
* A mirror to use as orientation reference when aligning camera.

## **Tips for Taking Good Photos for 4D24D**

I built 4D24D to be as easy as possible on the user, so you shouldn't have to worry too much about
getting perfect shots for your film scan. However, there are some things you can do to give yourself
a better chance of good results.

1. All shots MUST be in focus. 4D24D can correct for a lot, but if the film is not in focus, you will
    not get a good scan. Additionally, you may find that while the center of your shot is in focus, the
    edges are out of focus. Macro lenses are intended to compensate for this, and with a reversing ring
    you should be able to avoid the worst of it, but if you notice focus fall off, be sure you get at
    least one in-focus shot of every part of your film.
2. Do try to keep the camera as level as possible when shooting. This helps maintain focus, and saves
    4D24D from having to warp the image too much. Shooting significantly off level could potentially
    introduce distortion to your image, lessen the final scan quality or cause imperfect blending of
    image segments.
3. Keep your background clean. 4D24D should be able to pick out your image even without a strong
    backlight or film holder, but the more uniform the background is for your scans, the easier
    it will be for the algorithm to piece it all together.
