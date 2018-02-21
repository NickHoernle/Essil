
var BIOMES = ['Desert', 'Plains', 'Jungle', 'Wetlands', 'Reservoir'];
var BIOME_COLORS = ['#0100ff;', '#ff000a;', '#ff00f0;', '#00fff9;', '#fffb00;'];
var FILES = {
    '1': 'test1-2017-11-03T09-31.mp4',
    '2': 'test5-2017-11-16T11-41.mp4',
    '3': '',
    '4': '',
}

function pad(num, size) {
    var s = num+"";
    while (s.length < size) s = "0" + s;
    return s;
}

function update_time(event) {

    var vid = event.currentTarget;

    if (vid.currentTime < vid._startTime) {
        vid.currentTime = vid._startTime;
        vid.autoplay = true;
        vid.load()
    }
    if (vid.currentTime >= vid.end_time) {
        vid.pause();
    }
    var min = parseInt(vid.currentTime/60);
    var sec = parseInt(vid.currentTime - min*60);
    $('#time_stamp').html("<p><em>Time: " + pad(min,2) + ":" + pad(sec,2) + "</em></p>");
}

function seeking(vid) {
    vid._startTime = undefined;
}

function initialise_video(start_time, end_time) {

    var vid = document.getElementById("globalView");

    vid.removeEventListener('timeupdate', update_time);
    vid.removeEventListener('seeking', seeking);

    // start_time = start_time/6
    // end_time = end_time/6
    var speed = 6.0;

    // console.log(s/tart_time);
    vid._startTime = start_time;
    vid.end_time = end_time;

    vid.playbackRate = speed;
    vid.defaultPlaybackRate = speed;
    vid.loop = "loop";
    vid.autoplay = true;

    vid.addEventListener('timeupdate', update_time);
    vid.addEventListener('seeking', seeking);

    vid.currentTime = start_time;
    vid.play();
    vid.currentTime = start_time;
}

function make_description_list(list, description, ol) {

    if (list.length > 0) {

        var desc_ = document.createElement("p");
        var list_ = document.createElement(ol ? "ol" : 'ul');

        desc_.append(description);
        $(desc_).attr('class', 'coloured');

        $.each(list, function(k, item){

            var in_ = $('<li><b></b></li>');
            in_.text(" " + BIOMES[item]); //this is the value of the input
            in_.attr('class', 'water_in'); //use attr instead of setAttribute

            var span = $('<span></span>');
            span.attr('class', BIOMES[item].toLowerCase());

            in_.prepend(span);
            in_.appendTo(list_);

        });

        desc_.append(list_);
        return desc_;
    }
}

function list_events(events, text) {
    var event_txt = $('<div></div>');
    $.each(events, function(k, item){
        var evnt = $('<p><b></b></p>');
        evnt.text("** " + BIOMES[item] + (item == 4 ? " released" : text));
        evnt.css("font-weight","Bold");
        // if (item == 4) {
        //     evnt.css("color", "#fffb00");
        // }
        event_txt.append(evnt);
    });
    return event_txt;
}

function get_explanation_text(period, target) {
    target.empty();

    var header = "<h5>Period: " + period['number'] + "/" + period['num_periods'] + "</h5>";
    // var time_description = "<p><b>Start:</b> " + period['start'] + " | <b>End:</b> " + period['end'] + " | <b>Duration:</b> " + period['duration'] + 's</p>';
    target.append(header);

    // target.append(time_description);
    target.append(make_description_list(period['water_in'], 'Biomes receiving water:', true));
    target.append(make_description_list(period['no_change'], 'Biomes receiving little or no water:', false));

    target.append(list_events(period['rain'], ' has had a rain event.'));
    target.append(list_events(period['pipe_out'], ' has water piped out of biome.'));

    return true;
}

function render_explanation(periods, i) {

    var period = periods[i];
    get_explanation_text(period, $( "#explanation1" ));

    var period = periods[i+1];
    get_explanation_text(period, $( "#explanation2" ));
}


var current_index = 0;
var data_file = "";
var video_file = "";
var periods = [];

function round5(x) {
    return Math.ceil(x/5)*5;
}

function get_time_string(x) {
    var min = parseInt(x / 60);
    var sec = parseInt(x - min*60);
    return pad(min,2) + ":" + pad(sec,2);
}

function render_screen(current_index, periods) {

    if (periods.length > 0) {

        var period1 = periods[current_index];
        var period2 = periods[current_index + 1];

        // load the correct video
        // $('#globalView').attr('src', video_file);

        // render the correct portion of the video here
        initialise_video(period1["start_seconds"], period2["end_seconds"]);
        // $("#globalView")[0].load();

        // render the correct portion of the explanation
        render_explanation(periods, current_index);

        // $('#time_picker').empty();
        // $('#time_picker').
        // $('#time_picker').timepicker({
        //     'timeFormat': 'H:i',
        //     'step': 1,
        //     'minTime': period1.start + ':00',
        //     'maxTime': period2.end + ':00'
        // });
        $('#time_picker_td').empty();
        $('#time_picker_td').append("<span id='time_picker' class='ui-timepicker-input'></span> <button id='openSpanExample'>Pick Time</button>")
        $('#time_picker').timepicker(
            {
                'timeFormat': 'H:i',
                'step': 5,
                'minTime': get_time_string(round5(period1.start_seconds)) + ':00',
                'maxTime':  get_time_string(round5(period2.end_seconds)) + ':00'
            }
        );
        $('#openSpanExample').on('click', function(){
            $('#time_picker').timepicker('show');
        });
    }
}

$('#time_picker_td').on('selectTime', function(event){
    var time = $(event.target).timepicker('getTime');

    var seconds = time.getSeconds();
    var minutes = time.getMinutes();
    var hour = time.getHours();

    $('#time_picker_output').html("<p>" + pad(hour,2) + ":" + pad(minutes,2));
});

function handleFileSelect(evt) {

    // var files = evt.target; // FileList object
    var selected_file = FILES[evt.target.value];
    console.log(selected_file);

    data_file = './data/' + (selected_file.includes('.json') ? selected_file : selected_file.replace('.mp4', '.json'));
    video_file = './data/' + (selected_file.includes('.mp4') ? selected_file : selected_file.replace('.json', '.mp4'));

    // load the json data
    $.getJSON(data_file, function( data ) {

        console.log(data);
        periods = data;
        current_index = 0;

        // load the correct video
        var vid = document.getElementById("globalView");
        $('#globalView').attr('src', video_file);
        vid.load()

        render_screen(current_index, periods);
    });
}

$('#ex-basic').on('change', handleFileSelect);


// add button listener on the next_period button
$( "#next_period" ).click(function() {

    if (current_index >= periods[0]['num_periods']) {
        alert("You're at the last period");
    } else {
        current_index = current_index + 1;
        render_screen(current_index, periods);
    }
});

$( "#prev_period" ).click(function() {

    if (current_index <= 0) {
        alert("You're at the first period");
    } else {
        current_index = current_index - 1;
        render_screen(current_index, periods);
    }
});

$( "#replay_video" ).click(function() {

    render_screen(current_index, periods);
});


$('#ex-basic').trigger('change', '1');


