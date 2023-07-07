
html:
	export DEBUG=False && python3 app.py &
	sleep 10
	wget -r http://127.0.0.1:8050/
	wget -r http://127.0.0.1:8050/_dash-layout
	wget -r http://127.0.0.1:8050/_dash-dependencies
	wget -r http://127.0.0.1:8050/_reload-hash
	sed -i 's/_dash-layout/_dash-layout.json/g' 127.0.0.1:8050/_dash-component-suites/dash/dash-renderer/build/*.js
	sed -i 's/_dash-dependencies/_dash-dependencies.json/g' 127.0.0.1:8050/_dash-component-suites/dash/dash-renderer/build/*.js
	sed -i 's|<script src="/|<script src="|g' 127.0.0.1:8050/index.html
	sed -i 's|<script src="/|<script src="|g' 127.0.0.1:8050/robots.txt
	sed -i 's|href="/_favicon.ico?v=2.9.3|href="_favicon.ico|g' 127.0.0.1:8050/index.html
	sed -i 's|href="/_favicon.ico?v=2.9.3|href="_favicon.ico|g' 127.0.0.1:8050/robots.txt
	mv 127.0.0.1:8050/_dash-layout 127.0.0.1:8050/_dash-layout.json
	mv 127.0.0.1:8050/_dash-dependencies 127.0.0.1:8050/_dash-dependencies.json
	mv 127.0.0.1:8050/_favicon.ico?v=2.9.3 127.0.0.1:8050/_favicon.ico
	cp assets/* 127.0.0.1:8050/assets/
	cp _static/dcc/async* 127.0.0.1:8050/_dash-component-suites/dash/dcc/
	cp _static/table/async* 127.0.0.1:8050/_dash-component-suites/dash/dash_table/
	ps | grep python | awk '{print $$1}' | xargs kill -9


clean:
	rm -rf 127.0.0.1:8050/
	rm -rf joblib
	rm -rf predictions.pkl
	rm -rf modeling_short.html
