
$(document).ready(function() {
	
		$('.skillbar').each(function() {
				$(this).find('.skillbar-bar').animate({
					width: $(this).attr('data-percent')
				}, 3000);
		});
	
	
		$("input").change(function(){
	
			var reader = new FileReader();
			var filename = "";
	
			reader.onload = function(){
				$('#output').attr('src', reader.result);
	
				filename = $('input[type=file]').val().replace(/C:\\fakepath\\/i, '');
				$('#name-food').text();
			};
	
			reader.readAsDataURL(event.target.files[0]);		
			
			$('.skillbar').each(function() {
				$(this).find('.skillbar-bar').animate({
					width: $(this).attr('data-percent')
				}, 3000);
			});
			
		})
	
		
		
	});